# Responsible use

VAR-Offside is a research and learning tool. It is **not** a certified officiating system, and it isn't affiliated with or endorsed by FIFA, IFAB, any league or any broadcaster. "VAR" is used descriptively.

## Treat every call as an aid

- The decision depends on the detected players, the chosen frame and the calibration. Any of them can be wrong without the app knowing. Check the line-setting defender, the attacker and the pitch lines by eye.
- A single broadcast frame has limited resolution and the ball may have been played between frames. Margins of a few centimetres are below what this method can measure.
- Calibration has its own error, which grows towards the far side of the pitch and on wide angles.
- Auto-review answers come from a general vision language model with little measured accuracy on this task.

Please don't present its output as an official or definitive decision, especially about real people in public.

## Footage

Broadcast footage is usually owned by leagues and broadcasters. You're responsible for having the right to use the frames you analyse, and to share any images or videos you export. Auto-review sends frames to a third-party model provider; check that against your rights too.

Apart from a few app screenshots in `docs/assets/readme/` that illustrate the interface, the repository doesn't include broadcast footage, and pull requests that add some will be declined.

## Privacy

Frames show identifiable people. The app runs locally and keeps sessions in memory only; **Clear** deletes a frame and everything made from it, including exports on the server. Nothing leaves your machine unless you turn on Auto-review.

## Reporting a wrong call

Wrong calls are the most useful bug reports. Open an issue with the **Wrong call** template, and attach the frame only if you have the right to share it. Otherwise, describe it, or attach the exported `report.json`, which has no image.
