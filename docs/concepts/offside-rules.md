# Offside rules

VAR-Offside implements the **position** part of the offside law (IFAB Laws of the Game, Law 11): whether an attacker is nearer the opponents' goal line than the second-last opponent at the moment the ball is played. It doesn't judge whether the player was involved in play; that stays with the referee.

## The line

- The line is set by the **second-last defender**, counting the goalkeeper. In practice that's usually the last outfield defender.
- If you mark the keeper as **Not in frame**, the line is set by the **last visible defender**, because the keeper is assumed to be behind the camera's view.
- Only the defending team and the marked keeper count. Referees are never defenders.

## Which body parts count

The law counts any part of the head, body or feet, but not the hands and arms. The code follows that using the pose keypoints:

| Counts | Doesn't count |
|---|---|
| Head (nose, eyes, ears), shoulders, hips, knees, ankles | Elbows and wrists |

For both the defender and the attackers, the **most advanced** counting keypoint in the attack direction is used. A pose is only trusted when at least four torso and leg keypoints are visible and the points sit inside the player's box; otherwise the box's leading bottom corner is used. Each point is moved down to the row of the player's feet, so it sits on the ground, before it's projected onto the pitch.

## Deciding

| Situation | Result |
|---|---|
| Calibrated, attacker ahead by more than 0 m | Offside |
| Calibrated, attacker level or behind | Onside |
| Uncalibrated, attacker ahead by more than 3 px | Offside, marked **Estimate** |
| Attacker in their own half (on the halfway line counts as own half) | Onside |
| Something needed is missing (no defenders, no direction) | **No call**, with the reason |

The pixel tolerance is `offside.line_margin_px` in [`config.yaml`](../getting-started/configuration.md).

## What it doesn't do

- **Pick the moment.** The frame you give it is treated as the moment the ball is played. For a clip, you choose the frame.
- **Involvement in play**, interfering with an opponent, gaining an advantage, deliberate plays and the other judgement parts of Law 11.
- **The ball.** An attacker level with or behind the ball isn't treated specially.
- **Players out of frame.** A defender the camera can't see can't set the line. Broadcast angles often cut off the far side of the pitch.

## Accuracy

Very close calls can be smaller than the calibration's own error. A centimetre margin from one broadcast frame isn't a reliable decision. The pitch is assumed to be 105 × 68 m; set `offside.field_length` and `field_width` for other pitches. See [Responsible use](../about/responsible-use.md).
