import pyautogui
from PIL import Image
import google.generativeai as genai
import os
from pynput import keyboard, mouse
import threading
import logging
from typing import Tuple, Optional
import sys
import time
import random
import re

# C:/Users/rpsin/OneDrive/Desktop/Files/lifeLine/.venv/Scripts/Activate.ps1

# --- CONFIG ---
API_KEY = ""


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
input_suppression_active = False
input_suppression_listener = None
ctrl_pressed = False
alt_pressed = False

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
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

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

    logger.info("🖱️  Please click BOTTOM-RIGHT of question region...")
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
        "🖱️  Now click each option in order (click ALL options, then press Enter when done)"
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
    logger.info("🖱️  Click all options, then press Enter to continue...")
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
    logger.info("🖱️  Clicking correct option...")
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
            # Load all images
            images = [Image.open(path) for path in subjective_screenshots]

            # Enhanced prompt for better code formatting
            prompt = """
                    You are an expert problem solver.
                    For coding problems:
                    1. Always write the complete solution in Java inside a single public class named Solution
                    2. No indentation at all (the compiler will auto-indent)
                    3. No comments in the code
                    4. No extra spaces at the start of lines
                    5. Method bodies use standard Java braces style
                    6. No trailing whitespace at the end of lines
                    7. Line breaks only between logical code sections
                    8. Output only the code in plain text for direct copy-paste

                    For subjective or theoretical questions:
                    1. Give only the direct final answer
                    2. No explanations, no steps, no extra text
                    3. No comments or markdown formatting
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


def on_key_press(key):
    """Handle key press events for input suppression"""
    global ctrl_pressed, alt_pressed, is_typing_active, is_typing_paused

    # Update modifier state
    if (
        key == keyboard.Key.ctrl
        or key == keyboard.Key.ctrl_l
        or key == keyboard.Key.ctrl_r
    ):
        ctrl_pressed = True
    elif (
        key == keyboard.Key.alt
        or key == keyboard.Key.alt_l
        or key == keyboard.Key.alt_r
    ):
        alt_pressed = True
    elif (
        key == keyboard.KeyCode.from_char("z")
        and ctrl_pressed
        and alt_pressed
        and is_typing_active
    ):
        # Check for resume hotkey (Ctrl+Alt+Z)
        if is_typing_paused:
            resume_typing_event.set()
            logger.info("⏯️ Resume hotkey pressed. Resuming typing...")
        return True  # Allow the key to pass through

    # Block all other keys during active typing
    if is_typing_active and not is_typing_paused:
        return False

    return True


def on_key_release(key):
    """Handle key release events for input suppression"""
    global ctrl_pressed, alt_pressed

    # Update modifier state
    if (
        key == keyboard.Key.ctrl
        or key == keyboard.Key.ctrl_l
        or key == keyboard.Key.ctrl_r
    ):
        ctrl_pressed = False
    elif (
        key == keyboard.Key.alt
        or key == keyboard.Key.alt_l
        or key == keyboard.Key.alt_r
    ):
        alt_pressed = False


def start_input_suppression():
    """Start suppressing keyboard input"""
    global input_suppression_listener
    input_suppression_listener = keyboard.Listener(
        on_press=on_key_press, on_release=on_key_release
    )
    input_suppression_listener.start()
    logger.info("🔇 Input suppression activated")


def stop_input_suppression():
    """Stop suppressing keyboard input"""
    global input_suppression_listener
    if input_suppression_listener and input_suppression_listener.is_alive():
        input_suppression_listener.stop()
        logger.info("🔊 Input suppression deactivated")


def auto_type_response() -> None:
    """
    Types the stored AI response in a human-like way.
    Pauses if mouse click is detected (focus loss).
    Can be resumed from where it left off with resume hotkey.
    Triggered by Ctrl+Alt+T.
    """
    global ai_response_text, ai_response_ready_subjective
    global is_typing_active, is_typing_paused, current_typing_position, resume_typing_event

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

    logger.info("⌨️  Starting auto-type in 3 seconds. Click target field now...")
    time.sleep(3)

    # Start input suppression
    start_input_suppression()

    # Setup mouse listener to detect focus loss (mouse click)
    mouse_listener = None

    def on_mouse_click(x, y, button, pressed):
        # Use global for module-level variables
        global is_typing_paused, is_typing_active

        if pressed and button == mouse.Button.left and is_typing_active:
            is_typing_paused = True
            logger.info(
                "MouseClicked Mouse click detected. Pausing typing (focus lost)."
            )
            # Stop input suppression while paused (so user can type normally)
            stop_input_suppression()
            return False  # Stop the listener

    mouse_listener = mouse.Listener(on_click=on_mouse_click)
    mouse_listener.start()

    try:
        while current_typing_position < len(ai_response_text) and is_typing_active:
            if is_typing_paused:
                logger.info("⏸️ Typing paused. Press Ctrl+Alt+Z to resume...")
                resume_typing_event.wait()
                resume_typing_event.clear()
                is_typing_paused = False
                # Restart input suppression
                start_input_suppression()
                # Restart mouse listener after resuming
                mouse_listener = mouse.Listener(on_click=on_mouse_click)
                mouse_listener.start()
                continue

            # Type the next character
            char = ai_response_text[current_typing_position]
            pyautogui.write(char, interval=0.03)
            current_typing_position += 1

            # Add natural pauses - FIXED: Using random module correctly
            if ord(char) % 10 == 0:
                # Make sure to use the random module correctly
                time.sleep(random.uniform(0.1, 0.3))

        logger.info("✅ Finished typing AI response.")
    except Exception as e:
        logger.error(f"❌ Typing failed: {e}")
    finally:
        is_typing_active = False
        # Clean up mouse listener
        if mouse_listener and mouse_listener.running:
            mouse_listener.stop()
        # Stop input suppression
        stop_input_suppression()


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
        resume_typing_event.set()
        logger.info(
            f"⏯️ Resume hotkey pressed. Resuming typing from position {current_typing_position}..."
        )
    elif not is_typing_active:
        logger.info("⏯️ No active typing session to resume.")
    else:
        logger.info("⏯️ Typing is not currently paused.")


def on_clear_context() -> None:
    """Enhanced: Also clears subjective mode"""
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


def clear_context() -> None:
    """
    Clears the captured context.
    Triggered by Ctrl+Alt+R.
    """
    global context_captured
    context_captured = False
    logger.info("🧹 Context cleared. Future MCQs will NOT include context.")


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
        image = Image.open(image_path)
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
        context_img = Image.open(context_image_path)
        question_img = Image.open(question_image_path)

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


def on_clear_context() -> None:
    """Triggered by Ctrl+Alt+R"""
    logger.info("[🧹 CLEAR CONTEXT HOTKEY PRESSED!]")
    clear_context()


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

    with keyboard.GlobalHotKeys(
        {
            CAPTURE_CONTEXT_HOTKEY: on_activate_context,
            CAPTURE_QUESTION_HOTKEY: on_activate_question,
            CAPTURE_SUBJECTIVE_HOTKEY: on_capture_subjective,
            GENERATE_RESPONSE_HOTKEY: on_generate_response,
            TYPE_RESPONSE_HOTKEY: on_type_response,
            RESUME_TYPING_HOTKEY: on_resume_typing,  # New resume hotkey
            CLEAR_CONTEXT_HOTKEY: on_clear_context,
            EXIT_HOTKEY: on_exit,
        }
    ) as h:
        h.join()
