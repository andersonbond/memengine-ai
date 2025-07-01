import { XMarkIcon } from "@heroicons/react/24/solid";

export default function Modal({ isOpen, onClose, children }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded-xl shadow-lg relative w-96">
        <button
          className="absolute top-3 right-3 text-gray-500 hover:text-black"
          onClick={onClose}
        >
          <XMarkIcon className="w-5 h-5" />
        </button>
        {children}
      </div>
    </div>
  );
}
