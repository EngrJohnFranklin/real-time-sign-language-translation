"""
Real-Time Sign Language Translator Demo

Demonstrates the new real-time translation system with:
- Natural language output (e.g., "Hello" instead of "Shaka")
- Single current sign display (not accumulated)
- Duplicate detection (ignores held signs)
- Timeout-based clearing (clears after 3 seconds of inactivity)
- Smart TTS (only once per new sign)
"""

import time
from src.translation.realtime_translator import RealTimeTranslator, GESTURE_TO_MEANING


def demo_translator():
    """Demo the RealTimeTranslator with simulated sign detections."""
    translator = RealTimeTranslator()
    
    print("\n" + "="*70)
    print("REAL-TIME SIGN LANGUAGE TRANSLATOR DEMO")
    print("="*70)
    
    print("\n📊 Gesture to Meaning Mapping:")
    print("-" * 70)
    for gesture, meaning in sorted(GESTURE_TO_MEANING.items()):
        if meaning:
            print(f"  {gesture:20s} → {meaning}")
    
    print("\n\n🎬 Demo Scenario: User makes signs")
    print("-" * 70)
    
    # Scenario 1: New sign "Hello"
    print("\n[Scenario 1] User shows 'Hello' gesture")
    display, speak = translator.update("Hello", "Right", 0.85)
    print(f"  Display: '{display}'")
    print(f"  Speak: {speak} (confidence 0.85 ≥ 0.70)")
    assert display == "Hello", "Should show 'Hello'"
    assert speak == True, "Should speak on new high-confidence sign"
    
    # Scenario 2: Same sign held (duplicate detection)
    print("\n[Scenario 2] Same 'Hello' gesture held (user didn't move)")
    display, speak = translator.update("Hello", "Right", 0.88)
    print(f"  Display: {display} (None means no change)")
    print(f"  Speak: {speak} (no repeat TTS)")
    assert display is None, "Should not update on duplicate"
    assert speak == False, "Should not speak on duplicate"
    
    # Scenario 3: Different sign detected
    print("\n[Scenario 3] User changes to 'Thank You' gesture")
    display, speak = translator.update("Peace Sign", "Left", 0.82)
    print(f"  Display: '{display}'")
    print(f"  Speak: {speak} (new sign detected)")
    assert display == "Thank You", "Should show 'Thank You'"
    assert speak == True, "Should speak on new sign"
    
    # Scenario 4: Low confidence sign (below threshold)
    print("\n[Scenario 4] Low confidence detection - 'Yes' at 0.65 (below 0.70 threshold)")
    display, speak = translator.update("Thumbs Up", "Right", 0.65)
    print(f"  Display: '{display}'")
    print(f"  Speak: {speak} (confidence too low for speech)")
    assert display == "Yes", "Should show 'Yes'"
    assert speak == False, "Should NOT speak (confidence too low)"
    
    # Scenario 5: Timeout check
    print("\n[Scenario 5] Check timeout behavior")
    print(f"  Current display: '{translator.get_current_display()}'")
    print(f"  Current status: {translator.get_status()}")
    
    # Manually set last_update_time to simulate timeout
    translator.last_update_time = time.time() - 3.5  # 3.5 seconds ago
    timeout_result = translator.check_timeout()
    print(f"  After 3.5 seconds without new sign:")
    print(f"    Timeout triggered: {timeout_result == ''}")
    print(f"    Display cleared: {translator.get_current_display() == ''}")
    assert timeout_result == "", "Should return empty string on timeout"
    assert translator.get_current_display() == "", "Should clear display"
    
    print("\n\n✅ All demo scenarios passed!")
    print("="*70)


# def demo_gesture_mapping():
#     """Show all gesture to meaning mappings."""
#     print("\n" + "="*70)
#     print("GESTURE-TO-MEANING MAPPING REFERENCE")
#     print("="*70)
    
#     mappings = [
#         ("Closed Fist", "Sorry", "Hand fully closed"),
#         ("Open Palm", "Good Morning", "All fingers extended"),
#         ("Thumbs Up", "Yes", "Thumb extended upward"),
#         ("Thumbs Down", "No", "Thumb extended downward"),
#         ("Index Finger", "Help", "Only index finger extended"),
#         ("Peace Sign", "Thank You", "Index + middle extended"),
#         ("OK Sign", "Good", "Thumb + index in circle"),
#         ("I Love You", "I Love You", "Thumb, index, pinky extended"),
#         ("Hello", "Hello", "Thumb + pinky extended"),
#         ("Goodbye", "Goodbye", "All extended with middle-ring gap"),
#     ]
    
#     # print("\n{:<25} {:<25} {:<30}".format("GESTURE", "MEANING", "DESCRIPTION"))
#     # print("-" * 80)
#     # for gesture, meaning, description in mappings:
#     #     print("{:<25} {:<25} {:<30}".format(gesture, meaning, description))
    
#     # print("\n" + "="*70)


# if __name__ == "__main__":
#     demo_gesture_mapping()
#     print("\n")
#     demo_translator()
