import { expect, test } from "bun:test";
import { CallAudioSession } from "../frontend/audio/call-session.js";

test("call routing restores the previous audio mode, including across repeated calls", () => {
  const audioSession = { type: "playback" };
  for (let call = 0; call < 2; call++) {
    const owner = new CallAudioSession({ audioSession });
    expect(audioSession.type).toBe("play-and-record");
    expect(owner.routing).toBe("play-and-record");
    owner.close();
    expect(audioSession.type).toBe("playback");
    audioSession.type = "ambient";
    owner.close();
    expect(audioSession.type).toBe("ambient");
    audioSession.type = "playback";
  }
});

test("missing, malformed, rejected, and ignored routing APIs remain distinct", () => {
  for (const host of [{}, { audioSession: null }, { audioSession: { type: "unknown" } }]) {
    const owner = new CallAudioSession(host);
    expect(owner.routing).toBe("unavailable"); owner.close();
  }
  for (const host of [
    { get audioSession(): never { throw new Error("Unavailable"); } },
    { audioSession: { get type() { return "auto"; }, set type(_value: string) { throw new Error("Rejected"); } } },
    { audioSession: { get type() { return "auto"; }, set type(_value: string) {} } },
  ]) {
    const owner = new CallAudioSession(host);
    expect(owner.routing).toBe("not-applied"); owner.close();
  }
});

test("closing the call does not overwrite a newer audio mode selection", () => {
  const audioSession = { type: "auto" };
  const owner = new CallAudioSession({ audioSession });
  audioSession.type = "ambient";
  expect(owner.routing).toBe("not-applied");
  owner.close();
  expect(audioSession.type).toBe("ambient");
});
