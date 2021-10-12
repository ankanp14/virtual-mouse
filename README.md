## Virtual Mouse Using Hand Gestures

> Using this project we can control the mouse cursor movements using hand gestures.

##### Supports the following:
- Cursor movement
- Left click
- Drag/Long press

##### Dependencies:
- Python 3.7.9 (mediapipe does not work on higher versions)
- openCV
- mediapipe
- autopy
- numpy

##### Gestures:
- Move the index finger to control mouse cursor movement
- Touch the tip of the middle finger with the index finger to register a click
- Raise the little finger to register a drag or long press

### Instructions
In `VirtualMouse.py` set `deviceIndex` to either 0 or 1
> Run `VirtualMouse.py`