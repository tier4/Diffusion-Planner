import { useEffect, useState } from "react";
import type { PanelExtensionContext } from "@lichtblick/suite";

import { getState, initTopicStore, subscribe } from "./topicStore";
import type { AnnotationState } from "./types";

export function useWsState(context: PanelExtensionContext): AnnotationState {
  const [state, setState] = useState<AnnotationState>(getState());

  useEffect(() => {
    initTopicStore(context);
    const unsubscribe = subscribe(setState);
    return () => unsubscribe();
  }, [context]);

  return state;
}
