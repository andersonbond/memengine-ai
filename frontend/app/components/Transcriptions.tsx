import { useEffect, useRef, useState } from "react";
import { 
  TranscriptionSegment, 
  Participant,
  TrackPublication,
  RoomEvent, 
} from "livekit-client";
import { useMaybeRoomContext } from "@livekit/components-react";

export default function Transcriptions() {
  const room = useMaybeRoomContext();
  const [transcriptions, setTranscriptions] = useState<{ [id: string]: TranscriptionSegment }>({});
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!room) return;

    const updateTranscriptions = (
      segments: TranscriptionSegment[],
      participant?: Participant,
      publication?: TrackPublication
    ) => {
      setTranscriptions((prev) => {
        const newTranscriptions = { ...prev };
        for (const segment of segments) {
          newTranscriptions[segment.id] = segment;
        }
        return newTranscriptions;
      });
    };

    room.on(RoomEvent.TranscriptionReceived, updateTranscriptions);
    return () => {
      room.off(RoomEvent.TranscriptionReceived, updateTranscriptions);
    };
  }, [room]);

  // Scroll to bottom whenever the transcriptions update
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [transcriptions]);

  return (
    <div 
      ref={containerRef}
      className="px-4 bg-neutral-700 rounded-lg mx-2 mb-4 max-h-[200px] overflow-y-auto"
    >  
      <ul className="mt-2 leading-relaxed">
        {Object.values(transcriptions)
          .sort((a, b) => a.firstReceivedTime - b.firstReceivedTime)
          .map((segment) => (
            <li key={segment.id} className="mb-1">
              {segment.text}
            </li>
          ))}
      </ul>
    </div>
  );
}
