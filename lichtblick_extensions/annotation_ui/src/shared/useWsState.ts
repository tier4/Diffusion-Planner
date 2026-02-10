import { useEffect, useState } from "react";

import { ensureConnected, getState, subscribe } from "./wsClient";
import type { AnnotationState } from "./types";

export function useWsState(): AnnotationState {
  const [state, setState] = useState<AnnotationState>(getState());

  useEffect(() => {
    ensureConnected();
    const unsubscribe = subscribe(setState);
    return () => unsubscribe();
  }, []);

  return state;
}
