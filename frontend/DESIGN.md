# RokoSMS Design Context

## Product
RokoSMS is a Pakistan-focused Roman Urdu SMS/WhatsApp scam detector. The primary job of the interface is to let someone inspect a suspicious message quickly and understand the model's verdict and supporting signals.

## Visual direction
The interface is a quiet security instrument rather than a generic AI dashboard. It uses a near-black graphite field, warm paper-like text, restrained mint for legitimate results, and restrained coral-red for scam results. The result state is the signature: a soft ambient neon-like halo changes hue around the analysis surface, but never becomes a full-screen color wash or gradient.

## Typography
Use a compact geometric sans for interface text, with strong numeric/tabular emphasis for the probability. The product name is set with slightly expanded tracking and a small utility label. Avoid decorative display typography that would weaken a security-tool tone.

## Layout
Single-column analysis workspace with a narrow utility header and a message composer above the result. On wide screens, the composer/result relationship remains central and bounded rather than becoming a dashboard grid. On small screens, controls stack and result details remain readable without horizontal scrolling.

## Signature
The analysis surface has a soft state-reactive glow: red for scam, green for legitimate, neutral graphite before analysis. The transition is intentionally short and smooth enough to feel responsive.

## Interaction principles
- The primary action is always named "Check message".
- Loading preserves layout and disables duplicate submission.
- Verdict is communicated by text, icon, and semantic color.
- Explanations are shown as concrete signals, not as an opaque score explanation.
- URL analysis is separate from the model verdict and only appears when the API returns useful information.
- Errors explain the next action.
- Keyboard focus is visible; reduced-motion users get no state transition animation.
