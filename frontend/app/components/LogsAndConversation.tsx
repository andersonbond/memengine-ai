import React from "react";
import { useEffect, useState } from "react";

export function LogsAndConversations() {

  const [logs, setLogs] = useState<string[]>([]);
  // useEffect(() => {
  //   console.log('trigger logs...')
  //   const fetchLogs = async () => {
  //     const response = await fetch("http://127.0.0.1:8000/api/conversation");
  //     const data = await response.json();
  //     setLogs(data.logs);
  //   };
    
  //   // Poll the API every 5 seconds for new logs
  //   const interval = setInterval(fetchLogs, 5000);
  //   fetchLogs(); // Initial fetch
  //   return () => clearInterval(interval); // Cleanup
  // }, []);

  return (
    <div className="logs-container">
      <h2>Conversation:</h2>
      <ul>
        {logs.map((log, index) => (
          <li key={index}>{log}</li>
        ))}
      </ul>
    </div>
  );
}