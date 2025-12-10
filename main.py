import pyautogui
from PIL import Image
import google.generativeai as genai
import os
from pynput import keyboard, mouse
from pynput.keyboard import Controller as KeyboardController
import threading
import logging
from typing import Tuple, Optional
import sys
import time
import random
import re

# --- CONFIG ---
API_KEY_PRIMARY = ""
API_KEY_SECONDARY = ""

# API request management
REQUEST_LIMIT_BEFORE_SWITCH = 20  # Switch to secondary key after this many requests
api_request_count = 0
current_api_key = API_KEY_PRIMARY
api_key_lock = threading.Lock()

# Image compression settings
MAX_IMAGE_WIDTH = 1024  # Reduce image width to this max
MAX_IMAGE_HEIGHT = 1024  # Reduce image height to this max
IMAGE_QUALITY = 85  # JPEG quality (1-100)

SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# New: For coding/subjective mode
SUBJECTIVE_SCREENSHOT_DIR = os.path.join(SCREENSHOT_DIR, "subjective")
os.makedirs(SUBJECTIVE_SCREENSHOT_DIR, exist_ok=True)
# Paths for MCQ images (question + optional context)
QUESTION_IMAGE_NAME = "question.png"
CONTEXT_IMAGE_NAME = "context.png"
# Use absolute paths to avoid cwd issues when the script is invoked from other directories
QUESTION_IMAGE_PATH = os.path.join(os.path.abspath(SCREENSHOT_DIR), QUESTION_IMAGE_NAME)
CONTEXT_IMAGE_PATH = os.path.join(os.path.abspath(SCREENSHOT_DIR), CONTEXT_IMAGE_NAME)

# Hotkeys
CAPTURE_CONTEXT_HOTKEY = "<ctrl>+<alt>+c"
CAPTURE_QUESTION_HOTKEY = "<ctrl>+<alt>+p"
CLEAR_CONTEXT_HOTKEY = "<ctrl>+<alt>+r"
EXIT_HOTKEY = "<ctrl>+<alt>+e"

# NEW HOTKEYS
CAPTURE_SUBJECTIVE_HOTKEY = "<ctrl>+<alt>+s"  # Capture screenshot for coding
GENERATE_RESPONSE_HOTKEY = "<ctrl>+<alt>+g"  # Send to AI
TYPE_RESPONSE_HOTKEY = "<ctrl>+<alt>+t"  # Auto-type response
RESUME_TYPING_HOTKEY = "<ctrl>+<alt>+z"  # Resume typing after pause

# Global state
context_captured = False
subjective_screenshots = []  # List of paths: ['1.png', '2.png', ...]
ai_response_text = None  # Stores AI response for typing

# Typing state variables
is_typing_active = False
is_typing_paused = False
current_typing_position = 0
resume_typing_event = threading.Event()

# Set absolute log path
log_file_path = os.path.join(os.getcwd(), "mcq_automator.log")

# Clear existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
logger.propagate = True


# Optional: Log uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

# --- Initialize Gemini ---
genai.configure(api_key=current_api_key)
model = genai.GenerativeModel("gemini-2.5-flash")


def increment_api_request() -> None:
    """
    Increments the API request counter and switches to secondary API key if limit is reached.
    Thread-safe implementation.
    """
    global api_request_count, current_api_key, model

    with api_key_lock:
        api_request_count += 1
        logger.info(
            f"📊 API Request Count: {api_request_count}/{REQUEST_LIMIT_BEFORE_SWITCH}"
        )

        if (
            api_request_count >= REQUEST_LIMIT_BEFORE_SWITCH
            and current_api_key == API_KEY_PRIMARY
        ):
            current_api_key = API_KEY_SECONDARY
            genai.configure(api_key=current_api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            logger.warning(
                f"⚠️ REQUEST LIMIT REACHED ({REQUEST_LIMIT_BEFORE_SWITCH} requests)"
            )
            logger.warning("🔄 SWITCHED TO SECONDARY API KEY")
            logger.info("✅ Secondary API key configured successfully")


def compress_image(image_path: str) -> Image.Image:
    """
    Loads and compresses an image to reduce size before sending to Gemini.

    Args:
        image_path (str): Path to the original image

    Returns:
        PIL.Image.Image: Compressed image object
    """
    try:
        with Image.open(image_path) as image:
            original_size = os.path.getsize(image_path)

            # Calculate new dimensions while maintaining aspect ratio
            width, height = image.size
            if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
                ratio = min(MAX_IMAGE_WIDTH / width, MAX_IMAGE_HEIGHT / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(
                    f"🔧 Resized image from {width}x{height} to {new_width}x{new_height}"
                )

            # Convert to RGB if necessary (for JPEG compatibility)
            if image.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode == "P":
                    image = image.convert("RGBA")
                background.paste(
                    image,
                    mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None,
                )
                image = background

            # Create unique temp file to avoid conflicts
            import tempfile
            import uuid

            temp_filename = f"compressed_{uuid.uuid4().hex[:8]}.jpg"
            compressed_path = os.path.join(SCREENSHOT_DIR, temp_filename)

            image.save(compressed_path, "JPEG", quality=IMAGE_QUALITY, optimize=True)

        compressed_size = os.path.getsize(compressed_path)
        reduction = ((original_size - compressed_size) / original_size) * 100
        logger.info(
            f"📉 Image compressed: {original_size} bytes → {compressed_size} bytes ({reduction:.1f}% reduction)"
        )

        # Load compressed image and keep it in memory
        with Image.open(compressed_path) as compressed_img:
            # Create a copy to keep in memory
            result = compressed_img.copy()

        # Now safe to delete the temp file
        try:
            os.remove(compressed_path)
        except Exception as cleanup_error:
            # If deletion fails, log but don't crash - file will be cleaned up later
            logger.debug(f"⚠️ Temp file cleanup delayed: {cleanup_error}")

        return result

    except Exception as e:
        logger.error(f"❌ Image compression failed: {e}")
        # Return original image if compression fails
        return Image.open(image_path)


# Shared state for coordination
ai_response_ready = threading.Event()
option_positions_ready = threading.Event()
correct_index_global = None
option_positions_global = None
# Shared state for subjective mode
ai_response_ready_subjective = threading.Event()


def solve_current_mcq() -> None:
    """
    Captures MCQ question, sends to AI in background, and immediately starts capturing option positions.
    Auto-clicks once both AI response and option positions are ready.
    Works with any number of options.
    """
    global context_captured, ai_response_ready, option_positions_ready, correct_index_global, option_positions_global

    # Reset events and globals
    ai_response_ready.clear()
    option_positions_ready.clear()
    correct_index_global = None
    option_positions_global = None

    # --- Capture Question Region ---
    logger.info("❓ CAPTURE QUESTION: Please click TOP-LEFT of question region...")
    top_left = capture_single_click()
    if not top_left:
        logger.error("❌ Question capture cancelled.")
        return

    logger.info("MouseClicked Please click BOTTOM-RIGHT of question region...")
    bottom_right = capture_single_click()
    if not bottom_right:
        logger.error("❌ Question capture cancelled.")
        return

    x1, y1 = top_left
    x2, y2 = bottom_right
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        logger.error("❌ Invalid question region.")
        return

    try:
        screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
        screenshot.save(QUESTION_IMAGE_PATH)
        logger.info(f"✅ Question saved to {QUESTION_IMAGE_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to save question: {e}")
        return

    # --- Capture ALL Option Positions (any number) ---
    logger.info("\n" + "🟩" * 50)
    logger.info(
        "MouseClicked Now click each option in order (click ALL options, then press Enter when done)"
    )
    logger.info("🟩" * 50)

    option_positions = []
    done = threading.Event()

    def on_click(x, y, button, pressed):
        if pressed and button == mouse.Button.left:
            option_positions.append((x, y))
            logger.info(
                f"✅ Option {len(option_positions)} position recorded at ({x}, {y})"
            )
        return True

    def on_key_press(key):
        if key == keyboard.Key.enter:
            done.set()
            return False  # Stop listener
        return True

    # Start mouse and keyboard listeners
    mouse_listener = mouse.Listener(on_click=on_click)
    keyboard_listener = keyboard.Listener(on_press=on_key_press)

    mouse_listener.start()
    keyboard_listener.start()

    # Wait for user to indicate they're done
    logger.info("MouseClicked Click all options, then press Enter to continue...")
    done.wait(timeout=120)  # Wait up to 120 seconds

    # Clean up listeners
    mouse_listener.stop()
    keyboard_listener.stop()

    num_options = len(option_positions)
    if num_options < 2:
        logger.error(
            f"❌ Only {num_options} options captured. Need at least 2 options."
        )
        return

    logger.info(f"✅ Captured {num_options} options total")
    option_positions_global = option_positions
    option_positions_ready.set()

    # --- Start AI Request in Background Thread ---
    def ai_worker():
        global correct_index_global
        logger.info(f"🧠 Sending to Gemini (MCQ has {num_options} options)...")
        try:
            if context_captured:
                logger.info(f"📎 Including context... (MCQ has {num_options} options)")
                correct_index_global = get_correct_option_index_with_context(
                    CONTEXT_IMAGE_PATH, QUESTION_IMAGE_PATH, num_options
                )
            else:
                logger.info(
                    f"📎 NO context — sending question only... (MCQ has {num_options} options)"
                )
                correct_index_global = get_correct_option_index(
                    QUESTION_IMAGE_PATH, num_options
                )

            if correct_index_global is None:
                logger.error("❌ Failed to get answer from Gemini.")
            else:
                logger.info(f"✅ AI says correct option is: {correct_index_global}")
        except Exception as e:
            logger.error(f"❌ Error with Gemini: {e}")
        finally:
            ai_response_ready.set()
            os.remove(QUESTION_IMAGE_PATH)  # Clean up question image

    threading.Thread(target=ai_worker, daemon=True).start()

    # --- Wait for AI response ---
    logger.info("⏳ Waiting for AI response...")

    # Wait for AI response
    ai_response_ready.wait()

    # --- Validate Results ---
    if correct_index_global is None:
        logger.error("❌ No valid AI response received.")
        return

    if correct_index_global < 1 or correct_index_global > num_options:
        logger.error(
            f"❌ Invalid option index: {correct_index_global} (should be between 1 and {num_options})"
        )
        return

    # --- Click Correct Option ---
    logger.info("MouseClicked clicking correct option...")
    try:
        screen_x, screen_y = option_positions_global[correct_index_global - 1]
        pyautogui.moveTo(screen_x, screen_y, duration=0.3)
        pyautogui.click()
        logger.info(
            f"✅ Clicked option {correct_index_global} at ({screen_x}, {screen_y})"
        )
        logger.info("🎉 Done! Question answered.")
    except Exception as e:
        logger.error(f"❌ Click failed: {e}")


def capture_subjective_screenshot() -> None:
    """
    Captures a screenshot for coding/subjective question.
    Triggered by Ctrl+Alt+S.
    Can be called multiple times — stores screenshots in order.
    """
    global subjective_screenshots

    logger.info("🖼️  CAPTURE SUBJECTIVE SCREENSHOT: Please click TOP-LEFT...")
    top_left = capture_single_click()
    if not top_left:
        logger.error("❌ Capture cancelled.")
        return

    logger.info("MouseClicked Please click BOTTOM-RIGHT...")
    bottom_right = capture_single_click()
    if not bottom_right:
        logger.error("❌ Capture cancelled.")
        return

    x1, y1 = top_left
    x2, y2 = bottom_right
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        logger.error("❌ Invalid region.")
        return

    try:
        # Generate next filename
        idx = len(subjective_screenshots) + 1
        filename = f"{idx}.png"
        filepath = os.path.join(SUBJECTIVE_SCREENSHOT_DIR, filename)

        screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
        screenshot.save(filepath)
        subjective_screenshots.append(filepath)

        logger.info(f"✅ Saved screenshot {idx} to {filepath}")
        logger.info(f"📊 Total screenshots captured: {len(subjective_screenshots)}")
    except Exception as e:
        logger.error(f"❌ Failed to save screenshot: {e}")


def generate_ai_response() -> None:
    """
    Sends all captured subjective screenshots to Gemini and stores response.
    Runs in background — doesn't block hotkey system.
    """
    global subjective_screenshots, ai_response_text, ai_response_ready_subjective

    if not subjective_screenshots:
        logger.error("❌ No screenshots captured. Press Ctrl+Alt+S first.")
        return

    logger.info(f"🧠 Sending {len(subjective_screenshots)} screenshots to Gemini...")

    # Reset event
    ai_response_ready_subjective.clear()

    def ai_worker():
        global ai_response_text
        try:
            increment_api_request()

            # Load and compress all images
            images = [compress_image(path) for path in subjective_screenshots]
            prompt = """
You are taking an exam. You are an expert academic assistant who gives precise, rubric-following answers to subjective questions in an objective, professional academic style.

ANALYSIS PHASE:
1. Carefully analyze the question to identify ALL specified requirements, constraints, and rubrics
2. Note any specific formatting instructions, length requirements, or structural elements required
3. Identify key concepts, terminology, and depth of explanation needed
4. Determine if examples, calculations, or specific methodologies are required

RESPONSE PHASE:
1. Format your response EXACTLY as required by the question's rubrics - no deviations
2. Write in an objective, impersonal academic style
3. If the question specifies word count, paragraph structure, or section headings, follow them precisely
4. Include ONLY what is explicitly asked for - no introductions, conclusions, or extra explanations
5. Use the exact terminology and notation specified in the question
6. If calculations are required, show ONLY the final calculation format as specified (no working steps unless explicitly requested)
7. For theoretical questions, provide ONLY the direct answer without explanations
8. For coding problems only provide solution to the question in JAVA language, provide ONLY the code with no comments or extra text

COPY-PASTE READY FORMAT:
- NO markdown formatting
- NO introductory phrases ("The answer is:", "Here is the solution:", etc.)
- NO concluding statements
- NO bullet points or numbered lists unless explicitly required by the question
- NO extra line breaks beyond what's required by the rubrics
- NO comments in code solutions
- For coding: Only provide the complete, executable code with proper indentation as required by the language
- For essays: Follow the exact paragraph structure specified in the question

EXAMPLES OF PROPER RESPONSES:
For a coding question requiring a Java function:
"public class Solution {
    public static int calculateSum(int a, int b) {
        return a + b;
    }
}"

For a theoretical question asking for a definition:
"Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll."

For a question asking "Explain Newton's First Law":
"An object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force."

For a question specifying "Answer in 2 bullet points":
"- First point\n- Second point"

STRICT INSTRUCTIONS:
- NEVER add explanations, reasoning, or additional content beyond what's explicitly required
- If the question has multiple parts, address each part in the exact order specified
- If word count is specified, adhere to it precisely (e.g., "Answer in 50 words or less")
- Your response must be ready for immediate copy-pasting into the answer field without any editing
- Write in a way that feels authentic and human, avoiding detectable AI patterns like overly perfect grammar or repetitive phrasing
- Incorporate natural variations in language while maintaining academic accuracy
"""
            content = [prompt] + images
            response = model.generate_content(content)
            raw_response = response.text.strip()

            # Clean response to avoid triggering editor auto-indent
            cleaned_response = re.sub(r"\s+$", "", raw_response, flags=re.MULTILINE)
            cleaned_response = re.sub(r"\n{2,}", "\n", cleaned_response)

            ai_response_text = cleaned_response

            # Log the full AI response (not truncated)
            logger.info("✅ AI Response Received:")
            logger.info("=" * 50)
            logger.info("FULL AI RESPONSE:")
            logger.info("-" * 50)
            logger.info(ai_response_text)
            logger.info("-" * 50)
            logger.info(f"Response length: {len(ai_response_text)} characters")
            logger.info("=" * 50)
            logger.info("✅ Press Ctrl+Alt+T to auto-type this response.")

        except Exception as e:
            logger.error(f"❌ Error with Gemini: {e}")
            ai_response_text = None
        finally:
            ai_response_ready_subjective.set()

    # Run in background
    threading.Thread(target=ai_worker, daemon=True).start()
    logger.info("⏳ AI is working... You can press Ctrl+Alt+T later when ready.")


def auto_type_response() -> None:
    """
    Types the stored AI response with intelligent editor-aware indentation handling.
    Specifically handles code blocks to prevent double indentation.
    """
    global ai_response_text, is_typing_active, is_typing_paused, current_typing_position

    # Reset typing state
    is_typing_active = True
    is_typing_paused = False
    current_typing_position = 0

    if ai_response_text is None:
        logger.info("⏳ AI response not ready yet. Waiting...")
        ai_response_ready_subjective.wait(timeout=60)  # Wait max 60 sec
        if ai_response_text is None:
            logger.error("❌ AI response still not available after waiting.")
            is_typing_active = False
            return

    logger.info("⚠️ DO NOT TYPE OR CLICK DURING AUTO-TYPING! (Click will PAUSE typing)")
    logger.info("⌨️  Starting auto-type in 3 seconds. Click target field now...")
    time.sleep(3)

    # State tracking for code context
    brace_stack = []  # Tracks { and } for proper block detection
    current_indent = 0
    indent_size = 4  # Standard indentation size (spaces)
    i = 0
    mouse_listener = None

    # Mouse click handler for pause detection
    def on_mouse_click(x, y, button, pressed):
        nonlocal mouse_listener
        if pressed and button == mouse.Button.left and is_typing_active:
            # Set paused state
            global is_typing_paused
            is_typing_paused = True

            # Clean up current listener
            if mouse_listener and mouse_listener.is_alive():
                mouse_listener.stop()
                mouse_listener = None

            logger.info("⏸️ Mouse click detected. Pausing typing (focus loss).")
            return False  # Stop listener
        return True

    try:
        # Start mouse listener
        mouse_listener = mouse.Listener(on_click=on_mouse_click)
        mouse_listener.start()

        while i < len(ai_response_text) and is_typing_active:
            # Handle pause state
            if is_typing_paused:
                logger.info(
                    f"⏸️ Typing paused at position {i}. Press Ctrl+Alt+Z to resume..."
                )
                # Wait for resume signal
                resume_typing_event.wait()
                resume_typing_event.clear()

                # Reset pause state
                is_typing_paused = False

                # Restart mouse listener
                if mouse_listener and mouse_listener.is_alive():
                    mouse_listener.stop()
                mouse_listener = mouse.Listener(on_click=on_mouse_click)
                mouse_listener.start()

                # Release any held modifier keys
                keyboard_controller = KeyboardController()
                keyboard_controller.release(keyboard.Key.ctrl)
                keyboard_controller.release(keyboard.Key.alt)
                keyboard_controller.release(keyboard.Key.shift)
                time.sleep(0.1)

                continue

            char = ai_response_text[i]

            # Update code context state based on current character
            if char == "{":
                brace_stack.append(i)
            elif char == "}" and brace_stack:
                brace_stack.pop()

            # Special handling for newlines
            if char in ["\n", "\r"]:
                # Press Enter (editor will auto-indent)
                pyautogui.press("enter")
                time.sleep(0.05)  # Let editor apply auto-indent

                # Check if we're inside a code block (have unclosed braces)
                in_code_block = bool(brace_stack)

                # Skip AI's indentation if editor already provided it
                if in_code_block:
                    # Count leading spaces in AI response for next line
                    j = i + 1
                    ai_indent = 0
                    while j < len(ai_response_text) and ai_response_text[j] == " ":
                        ai_indent += 1
                        j += 1

                    # Skip spaces if editor already indented
                    if ai_indent > 0:
                        logger.debug(f"⏭️ Skipping {ai_indent} auto-indent spaces")
                        i = j - 1  # Skip to after indentation
                else:
                    # Outside code blocks - just skip any leading spaces
                    j = i + 1
                    while j < len(ai_response_text) and ai_response_text[j] == " ":
                        j += 1
                    i = j - 1

                i += 1  # Move past newline
                continue

            # Handle regular characters
            pyautogui.write(char, interval=random.uniform(0.03, 0.08))
            i += 1
            current_typing_position = i

            # Natural typing rhythm
            if i % 15 == 0:
                time.sleep(random.uniform(0.05, 0.15))

        logger.info("✅ Finished typing AI response.")
    except Exception as e:
        logger.error(f"❌ Typing failed: {e}")
    finally:
        is_typing_active = False
        # Clean up mouse listener
        if mouse_listener and mouse_listener.is_alive():
            mouse_listener.stop()


def on_capture_subjective() -> None:
    """Triggered by Ctrl+Alt+S"""
    logger.info("[🖼️  SUBJECTIVE CAPTURE HOTKEY PRESSED!]")
    threading.Thread(target=capture_subjective_screenshot).start()


def on_generate_response() -> None:
    """Triggered by Ctrl+Alt+G"""
    logger.info("[🧠 GENERATE RESPONSE HOTKEY PRESSED!]")
    threading.Thread(target=generate_ai_response, daemon=True).start()


def on_type_response() -> None:
    """Triggered by Ctrl+Alt+T"""
    logger.info("[⌨️  TYPE RESPONSE HOTKEY PRESSED!]")
    threading.Thread(target=auto_type_response).start()


def on_resume_typing() -> None:
    """Triggered by Ctrl+Alt+Z to resume paused typing"""
    global is_typing_paused, resume_typing_event

    if is_typing_paused and is_typing_active:
        # Release any held modifier keys before resuming
        keyboard_controller = KeyboardController()
        keyboard_controller.release(keyboard.Key.ctrl)
        keyboard_controller.release(keyboard.Key.alt)
        keyboard_controller.release(keyboard.Key.shift)
        time.sleep(0.1)

        resume_typing_event.set()
        logger.info(
            f"⏯️ Resume hotkey pressed. Resuming typing from position {current_typing_position}..."
        )
        time.sleep(0.2)  # Prevent immediate re-pause from mouse movement
    elif not is_typing_active:
        logger.info("⏯️ No active typing session to resume.")
    else:
        logger.info("⏯️ Typing is not currently paused.")


def on_clear_context() -> None:
    """
    Enhanced: Also clears subjective mode
    Clears the captured context.
    Triggered by Ctrl+Alt+R.
    """
    global context_captured, subjective_screenshots, ai_response_text
    context_captured = False
    subjective_screenshots = []
    ai_response_text = None
    # Clear files
    for file in os.listdir(SUBJECTIVE_SCREENSHOT_DIR):
        os.remove(os.path.join(SUBJECTIVE_SCREENSHOT_DIR, file))
    logger.info("🧹 Context & subjective mode cleared.")


def capture_context() -> None:
    """
    Captures context passage region and saves it.
    Triggered by Ctrl+Alt+C.
    """
    global context_captured
    logger.info("📖 CAPTURE CONTEXT: Please click TOP-LEFT of context region...")
    top_left = capture_single_click()
    if not top_left:
        logger.error("❌ Context capture cancelled.")
        return

    logger.info("MouseClicked Please click BOTTOM-RIGHT of context region...")
    bottom_right = capture_single_click()
    if not bottom_right:
        logger.error("❌ Context capture cancelled.")
        return

    x1, y1 = top_left
    x2, y2 = bottom_right
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        logger.error("❌ Invalid context region.")
        return

    try:
        screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
        screenshot.save(CONTEXT_IMAGE_PATH)
        logger.info(f"✅ Context saved to {CONTEXT_IMAGE_PATH}")
        context_captured = True
        logger.info("✅ Context is now ACTIVE. Next MCQ will include it.")
    except Exception as e:
        logger.error(f"❌ Failed to save context: {e}")


def get_correct_option_index(image_path: str, num_options: int = 4) -> Optional[int]:
    """
    Single-image mode — for standalone MCQs

    Sends an MCQ screenshot to Google Gemini Vision AI and returns the correct option index.

    Args:
        image_path (str): Path to the screenshot image file.
        num_options (int): Number of options in the MCQ.

    Returns:
        int or None: The correct option number (1 to num_options) if successfully parsed.
                     Returns None if response is invalid or parsing fails.

    Purpose:
        Uses multimodal AI to analyze the MCQ image and return the index of the correct answer.
        Assumes the question has exactly num_options vertically listed options.

    Example:
        If AI responds with "3", returns int(3).
        If AI responds with "Option 3" or "third", parsing will fail (prompt ensures it won't).

    Note:
        The prompt is designed to force AI to respond with ONLY a single digit (1 to num_options).
    """
    try:
        increment_api_request()

        image = compress_image(image_path)
        prompt = f"""
        You are an expert in answering multiple choice questions.
        The image shows a question with {num_options} vertically listed options.
        Return ONLY the number (1, 2, ..., {num_options}) of the correct option.
        Do NOT explain. Do NOT add any text.
        """
        response = model.generate_content([prompt, image])
        answer_text = response.text.strip()
        logger.info(f"🤖 Gemini says: {answer_text}")

        index = int(answer_text)
        if 1 <= index <= num_options:
            return index
        else:
            logger.error(f"❌ Invalid option index: {index}")
            return None
    except Exception as e:
        logger.error(f"❌ Invalid response from Gemini: {e}")
        return None


def get_correct_option_index_with_context(
    context_image_path: str, question_image_path: str, num_options: int = 4
) -> Optional[int]:
    """Two-image mode — for comprehension-based MCQs."""
    try:
        increment_api_request()

        context_img = compress_image(context_image_path)
        question_img = compress_image(question_image_path)

        prompt = f"""
        You are an expert in answering comprehension-based multiple choice questions.
        The FIRST image contains the context/passage.
        The SECOND image contains the question and {num_options} options.
        Read the context carefully, then answer the question.
        Return ONLY the number (1, 2, ..., {num_options}) of the correct option.
        Do NOT explain. Do NOT add any text.
        """

        response = model.generate_content([prompt, context_img, question_img])
        answer_text = response.text.strip()
        logger.info(f"🤖 Gemini says: {answer_text}")

        index = int(answer_text)
        if 1 <= index <= num_options:
            return index
        else:
            logger.error(f"❌ Invalid option index: {index}")
            return None
    except Exception as e:
        logger.error(f"❌ Invalid response from Gemini: {e}")
        return None


def capture_single_click() -> Optional[Tuple[int, int]]:
    """
    Waits for the user to perform a single left mouse click and returns its screen coordinates.

    Returns:
        tuple(int, int) or None: (x, y) coordinates of the click, or None if interrupted.

    Purpose:
        Used to capture dynamic screen positions from user input.
        Essential for:
          - Defining screenshot region (top-left, bottom-right)
          - Recording option positions (option 1, 2, 3, 4)

    Behavior:
        Blocks until a left mouse click is detected.
        Returns immediately after the first click.

    Example:
        User clicks at (500, 300) → returns (500, 300)
    """
    position = None
    clicked = threading.Event()

    def on_click(x: int, y: int, button: mouse.Button, pressed: bool):
        nonlocal position
        if pressed and button == mouse.Button.left:
            position = (x, y)
            clicked.set()
            return False  # Stop listener

    with mouse.Listener(on_click=on_click) as listener:
        clicked.wait()  # Wait for click

    return position


def on_activate_context() -> None:
    """Triggered by Ctrl+Alt+C"""
    logger.info("[📖 CONTEXT CAPTURE HOTKEY PRESSED!]")
    threading.Thread(target=capture_context).start()


def on_activate_question() -> None:
    """Triggered by Ctrl+Alt+P"""
    logger.info("[❓ QUESTION CAPTURE HOTKEY PRESSED!]")
    threading.Thread(target=solve_current_mcq).start()


def on_exit() -> None:
    """Triggered by Ctrl+Alt+E"""
    logger.info("🛑 Exiting...")
    os._exit(0)


# --- Main Listener ---
if __name__ == "__main__":
    logger.info("✅ Advanced AI Assistant Ready!")
    logger.info(
        f"📖 Press {CAPTURE_CONTEXT_HOTKEY.upper()} to capture context (for MCQs)"
    )
    logger.info(
        f"❓ Press {CAPTURE_QUESTION_HOTKEY.upper()} to capture question (for MCQs)"
    )
    logger.info(
        f"🖼️  Press {CAPTURE_SUBJECTIVE_HOTKEY.upper()} to capture screenshot (for coding)"
    )
    logger.info(f"🧠 Press {GENERATE_RESPONSE_HOTKEY.upper()} to generate AI response")
    logger.info(f"⌨️  Press {TYPE_RESPONSE_HOTKEY.upper()} to auto-type response")
    logger.info(f"⏯️  Press {RESUME_TYPING_HOTKEY.upper()} to resume typing after pause")
    logger.info(f"🧹 Press {CLEAR_CONTEXT_HOTKEY.upper()} to clear everything")
    logger.info(f"🚪 Press {EXIT_HOTKEY.upper()} to quit")
    logger.info(
        "⚠️ IMPORTANT: During auto-typing, DO NOT TYPE OR CLICK (click will pause typing)"
    )

    with keyboard.GlobalHotKeys(
        {
            CAPTURE_CONTEXT_HOTKEY: on_activate_context,
            CAPTURE_QUESTION_HOTKEY: on_activate_question,
            CAPTURE_SUBJECTIVE_HOTKEY: on_capture_subjective,
            GENERATE_RESPONSE_HOTKEY: on_generate_response,
            TYPE_RESPONSE_HOTKEY: on_type_response,
            RESUME_TYPING_HOTKEY: on_resume_typing,
            CLEAR_CONTEXT_HOTKEY: on_clear_context,
            EXIT_HOTKEY: on_exit,
        }
    ) as h:
        h.join()
