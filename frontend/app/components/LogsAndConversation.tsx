import React from "react";

interface LogsAndConversationsProps {
  items: { type: "log" | "conversation"; content: string }[];
}

export function LogsAndConversations({ items }: LogsAndConversationsProps) {
  return (
    <div className="bg-white p-4 shadow-md rounded-md max-h-[300px] overflow-y-auto">
      <h2 className="text-lg font-semibold mb-2">Logs & Conversations</h2>
      {items.length > 0 ? (
        items.map((item, index) => (
          <div
            key={index}
            className={`mb-2 p-2 rounded-md ${
              item.type === "log"
                ? "bg-gray-100 text-gray-800"
                : "bg-blue-100 text-blue-800"
            }`}
          >
            <p>
              <strong>{item.type === "log" ? "Log:" : "Conversation:"}</strong>{" "}
              {item.content}
            </p>
          </div>
        ))
      ) : (
        <p>No logs or conversations yet.</p>
      )}
    </div>
  );
}