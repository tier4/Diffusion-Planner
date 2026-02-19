import type { WsMessage } from "./types";
import { publishCommand } from "./topicStore";

export function sendMessage(message: WsMessage): void {
  publishCommand(message);
}
