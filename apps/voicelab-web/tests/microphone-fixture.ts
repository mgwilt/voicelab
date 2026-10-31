export function microphone(
  settings: MediaTrackSettings = {},
  apply: (constraints?: MediaTrackConstraints) => Promise<void> = async () => {},
  constraints: MediaTrackConstraints = {},
): MediaStreamTrack {
  let state: MediaStreamTrackState = "live";
  return {
    kind: "audio", id: "mic", label: "Microphone", enabled: true, muted: false,
    contentHint: "speech", get readyState() { return state; },
    onended: null, onmute: null, onunmute: null,
    stop() { state = "ended"; }, clone: () => microphone(settings, apply, constraints),
    applyConstraints: apply, getCapabilities: () => ({}),
    getConstraints: () => constraints, getSettings: () => settings,
    addEventListener: () => {}, removeEventListener: () => {}, dispatchEvent: () => true,
  };
}
