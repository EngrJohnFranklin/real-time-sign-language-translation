"""
Main Entry Point for Real-Time Sign Language Translation System.

This is the primary entry point for the application. It handles:
- Environment setup and validation
- Dependency checking
- Application initialization
- Error handling and logging
- GUI application launch

To run the application:
    python main.py
    
or:
    python -m src.main
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import List, Tuple

# Configure logging before any other imports
def _setup_logging():
    """
    Configure logging for the application.
    
    Sets up both file and console logging with appropriate levels and formats.
    """
    try:
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Log file path
        log_file = log_dir / "sign_language_app.log"
        
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        
        # File handler (DEBUG level)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        # Console handler (INFO level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        logger.info("Logging configured successfully")
        return logger
    
    except Exception as e:
        print(f"Error configuring logging: {e}")
        # Create basic console logging as fallback
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger()


# Setup logging immediately
logger = _setup_logging()


def _get_project_root() -> Path:
    """
    Get the project root directory.
    
    Walks up the directory tree to find the project root.
    
    Returns:
        Path to project root directory.
    """
    try:
        # Current file is src/main.py, so parent.parent is project root
        current_file = Path(__file__).resolve()
        
        # Try to find project root by looking for setup.py or pyproject.toml
        current_dir = current_file.parent
        for _ in range(5):  # Check up to 5 levels
            if any(marker in os.listdir(current_dir) for marker in ['setup.py', 'pyproject.toml', 'requirements.txt']):
                logger.debug(f"Project root found at: {current_dir}")
                return current_dir
            current_dir = current_dir.parent
        
        # Fallback to parent of src directory
        src_dir = current_file.parent
        project_root = src_dir.parent
        logger.debug(f"Project root assumed at: {project_root}")
        return project_root
    
    except Exception as e:
        logger.error(f"Error determining project root: {e}")
        return Path.cwd()


def _check_python_version() -> bool:
    """
    Check if Python version meets requirements.
    
    Returns:
        True if Python version is sufficient, False otherwise.
    """
    try:
        required_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= required_version:
            logger.info(f"Python version check passed: {sys.version.split()[0]}")
            return True
        else:
            logger.error(f"Python {required_version[0]}.{required_version[1]}+ required, got {current_version[0]}.{current_version[1]}")
            return False
    
    except Exception as e:
        logger.error(f"Error checking Python version: {e}")
        return False


def _check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if all required dependencies are installed.
    
    Returns:
        Tuple of (all_installed: bool, missing_packages: List[str])
    """
    required_packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'customtkinter': 'customtkinter',
        'vosk': 'vosk',
        'pyttsx3': 'pyttsx3',
        'PIL': 'Pillow',
        'numpy': 'numpy',
        'pyaudio': 'pyaudio',
    }
    
    missing = []
    
    try:
        logger.info("Checking dependencies...")
        
        for import_name, package_name in required_packages.items():
            try:
                __import__(import_name)
                logger.debug(f"[+] {package_name} installed")
            except ImportError:
                logger.warning(f"[-] {package_name} not installed")
                missing.append(package_name)
        
        if missing:
            logger.error(f"Missing packages: {', '.join(missing)}")
            return False, missing
        else:
            logger.info("All dependencies are installed")
            return True, []
    
    except Exception as e:
        logger.error(f"Error checking dependencies: {e}")
        return False, list(required_packages.values())


def _setup_path() -> bool:
    """
    Setup Python path for module imports.
    
    Adds src directory to sys.path for proper module resolution.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        project_root = _get_project_root()
        src_dir = project_root / "src"
        
        if src_dir not in sys.path:
            sys.path.insert(0, str(src_dir))
            logger.debug(f"Added to sys.path: {src_dir}")
        
        if project_root not in sys.path:
            sys.path.insert(0, str(project_root))
            logger.debug(f"Added to sys.path: {project_root}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error setting up path: {e}")
        return False


def _validate_resources() -> bool:
    """
    Validate that required resource directories exist or can be created.
    
    Returns:
        True if resources are valid or can be created, False otherwise.
    """
    try:
        project_root = _get_project_root()
        
        # Check/create required directories
        required_dirs = [
            project_root / "logs",
            project_root / "config",
            project_root / "data",
        ]
        
        for dir_path in required_dirs:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Directory ready: {dir_path}")
            except Exception as e:
                logger.warning(f"Could not create/access directory {dir_path}: {e}")
        
        # Check for videos directory (optional but recommended)
        videos_dir = project_root / "videos"
        if videos_dir.exists():
            logger.info(f"Videos directory found at: {videos_dir}")
        else:
            logger.warning(f"Videos directory not found at: {videos_dir}")
            logger.warning("Sign language demonstration videos will not be available")
        
        return True
    
    except Exception as e:
        logger.error(f"Error validating resources: {e}")
        return False


def _show_startup_banner():
    """Display startup banner with system information."""
    try:
        logger.info("=" * 70)
        logger.info("Real-Time Sign Language Translation System")
        logger.info("=" * 70)
        logger.info(f"Python Version: {sys.version.split()[0]}")
        logger.info(f"Platform: {sys.platform}")
        logger.info(f"Project Root: {_get_project_root()}")
        logger.info("=" * 70)
    except Exception as e:
        logger.warning(f"Error showing startup banner: {e}")


def _initialize_application():
    """
    Initialize the application before launching GUI.
    
    Performs all setup tasks including path setup, dependency checking,
    and resource validation.
    
    Returns:
        True if initialization successful, False otherwise.
    """
    try:
        logger.info("Initializing application...")
        
        # Show startup banner
        _show_startup_banner()
        
        # Check Python version
        logger.info("Checking Python version...")
        if not _check_python_version():
            logger.error("Python version check failed")
            return False
        
        # Check dependencies
        logger.info("Checking dependencies...")
        deps_ok, missing_packages = _check_dependencies()
        if not deps_ok:
            logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            logger.error("Please install missing packages using:")
            logger.error("  pip install " + " ".join(missing_packages))
            return False
        
        # Setup path
        logger.info("Setting up Python path...")
        if not _setup_path():
            logger.error("Failed to setup Python path")
            return False
        
        # Validate resources
        logger.info("Validating resources...")
        if not _validate_resources():
            logger.warning("Resource validation had some issues, continuing anyway...")
        
        logger.info("Application initialization completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error during application initialization: {e}")
        logger.error(traceback.format_exc())
        return False


def _launch_gui():
    """
    Launch the CustomTkinter GUI application.
    
    Imports and runs the main GUI window. This is done in a separate
    function to allow for proper error handling.
    
    Returns:
        True if GUI exited normally, False otherwise.
    """
    try:
        logger.info("Importing GUI module...")
        from ui.gui import MainWindow
        
        logger.info("Creating main window...")
        app = MainWindow()
        
        logger.info("Launching application window...")
        app.mainloop()
        
        logger.info("Application closed normally")
        return True
    
    except ImportError as e:
        logger.error(f"Failed to import GUI module: {e}")
        logger.error("Make sure you're running from the project root directory")
        return False
    
    except Exception as e:
        logger.error(f"Error launching GUI: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    """
    Main entry point for the application.
    
    Orchestrates initialization, validation, and GUI launch with
    comprehensive error handling.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        # Initialize application
        if not _initialize_application():
            logger.error("Application initialization failed")
            print("\n❌ Failed to initialize application")
            print("Please check the logs/sign_language_app.log file for details")
            return 1
        
        # Launch GUI
        logger.info("Launching GUI application...")
        if _launch_gui():
            logger.info("Application exited successfully")
            print("\n✓ Application closed successfully")
            return 0
        else:
            logger.error("GUI launch failed")
            print("\n❌ Failed to launch GUI application")
            print("Please check the logs/sign_language_app.log file for details")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C)")
        print("\n✓ Application terminated by user")
        return 0
    
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        logger.error(traceback.format_exc())
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the logs/sign_language_app.log file for details")
        return 1


if __name__ == "__main__":
    """
    Application entry point.
    
    This block ensures the application only runs when executed directly,
    not when imported as a module.
    """
    exit_code = main()
    sys.exit(exit_code)
