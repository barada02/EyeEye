# In Your Eyes

A journey drawn entirely with text characters. Look into an eye, fall through its pupil into a galaxy, toward one small star and its planets, the Earth, a single light on the night side, the home it belongs to, and the heart inside. Then it begins again.

```
eye → galaxy → solar system → Earth → home → love → eye …
```

## Run it

```
pip install -r requirements.txt
python eye.py
```

On Windows you can also use `py -3 eye.py`. You need Python 3 with `numpy` and `pygame-ce`.

## Controls

| Key / input      | What it does                                        |
|------------------|-----------------------------------------------------|
| Mouse            | The eye follows the cursor                          |
| Hold left click  | Shine a light: the pupil constricts                 |
| Space            | Travel deeper, into the next world                  |
| Backspace        | Travel back out to the previous world               |
| A                | Autoplay the whole journey (good for recording)     |
| B                | Blink                                               |
| + / -            | Bigger / smaller characters                         |
| G                | Toggle the background glow                          |
| S                | Save a screenshot (`eye_<date>_<time>.png`)         |
| H                | Show / hide the help line                           |
| F11              | Fullscreen                                          |
| Esc              | Quit (Ctrl+C in the terminal also works)            |

## The worlds

1. **Eye**: an iris of letters, lashes and brows drawn as strokes. It blinks, wanders and reacts to light, and holds a faint galaxy in its pupil.
2. **Galaxy**: a spiral galaxy with one warm star out on an arm.
3. **Solar system**: a sun with planets on dotted orbits. The Earth is ringed.
4. **Earth**: a rotating globe with continents, clouds and city lights on the night side.
5. **Home**: a house at night whose walls are written in "home".
6. **Love**: a beating heart written in "love". An eye opens at its centre, and the journey starts again.

## Files

- `eye.py`: the app, all the worlds and the text renderer
- `earth_map.py`: rough continent outlines for the globe
- `requirements.txt`: dependencies
