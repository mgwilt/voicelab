import type { ControlPanel } from "../components/panel.js";
import type { Model } from "../domain/experiment.js";
import { unreachable } from "../domain/experiment.js";
import { BreezePanel } from "./breeze/panel.js";
import { DuplexPanel } from "./nemotron/panel.js";
import { SynthesisPanel } from "./sesame/panel.js";

import { createPanel as phononPanel } from "./phonon/panel.js";
import { createPanel as kokoroPanel } from "./kokoro/panel.js";
import { createPanel as audio8Panel } from "./audio8/panel.js";
import { createPanel as ming_ttsPanel } from "./ming_tts/panel.js";
import { createPanel as dramaPanel } from "./drama/panel.js";
import { createPanel as mayaPanel } from "./maya/panel.js";
import { createPanel as omnivoicePanel } from "./omnivoice/panel.js";
import { createPanel as kittenPanel } from "./kitten/panel.js";
import { PersonaPlexPanel } from "./personaplex/panel.js";
import { MingAudioPanel } from "./ming_audio/panel.js";
import { QwenOmniPanel } from "./qwen_omni/panel.js";
import { MiniCPMPanel } from "./minicpm/panel.js";

export function createPanel(model: Model): ControlPanel {
  switch (model.id) {
    case "nemotron": return new DuplexPanel(model);
    case "sesame": return new SynthesisPanel(model);
    case "breeze": return new BreezePanel(model);
    case "phonon": return phononPanel(model);
    case "kokoro": return kokoroPanel(model);
    case "audio8": return audio8Panel(model);
    case "ming_tts": return ming_ttsPanel(model);
    case "drama": return dramaPanel(model);
    case "maya": return mayaPanel(model);
    case "omnivoice": return omnivoicePanel(model);
    case "kitten": return kittenPanel(model);
    case "personaplex": return new PersonaPlexPanel(model);
    case "ming_audio": return new MingAudioPanel(model);
    case "qwen_omni": return new QwenOmniPanel(model);
    case "minicpm": return new MiniCPMPanel(model);
    default: return unreachable(model);
  }
}
