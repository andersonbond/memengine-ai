"use client"
//import { PhoneIcon } from '@heroicons/react/24/solid';
import { useRouter } from 'next/navigation';
export default function Home() {
    const router = useRouter();
 const handleClick = () => {
    router.push('/pages/callpage')
 }
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-blue-50 to-blue-100 text-gray-800">
      <div className="text-center mb-8">
        <h1 className="text-5xl font-extrabold tracking-tight mb-4 text-gray-900">
          Welcome to <span className="text-[#ffb703]"><u>SGV AI</u></span>
        </h1>
      </div>

      <div className="flex items-center space-x-4 mt-8">
        <button 
            className="flex items-center px-6 py-3 bg-[#ffb703] hover:bg-[#f8c915] text-black rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition"
            onClick={handleClick}
        >
   
            Get Started
        </button>
      </div>


      <footer className="absolute bottom-4 text-sm text-gray-500">
        © {new Date().getFullYear()} SGV AI. All rights reserved.
      </footer>
    </div>
  );
}
