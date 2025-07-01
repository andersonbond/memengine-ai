import { useEffect } from "react";

export default function Notification({ message, type = "info", onClose }: any) {
  // Auto close after 3 seconds
  useEffect(() => {
    if (!message) return;
    const timer = setTimeout(() => {
      onClose();
    }, 3000);
    return () => clearTimeout(timer);
  }, [message, onClose]);

  if (!message) return null;

  const bgColor =
    type === "success"
      ? "bg-green-500"
      : type === "error"
      ? "bg-red-500"
      : "bg-blue-500";

  return (
    <div
      className={`fixed top-4 right-4 px-6 py-3 rounded shadow-md text-white font-semibold ${bgColor}`}
      role="alert"
    >
      {message}
    </div>
  );
}
