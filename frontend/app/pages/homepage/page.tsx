"use client";
//import Image from 'next/image';
import { PhoneIcon } from "@heroicons/react/24/solid";
import { useRouter } from "next/navigation";
//import logo from '@/app/assets/logo_ai.png';

export default function Home() {
  const router = useRouter();

  const handleClick = () => {
    router.push("/pages/callpage");
  };

  return (
    <div className="flex p-4 flex-col items-center justify-center min-h-screen bg-gradient-to-b from-blue-50 to-blue-100 text-gray-800">
      <div className="text-center mb-8 flex flex-col items-center">
        {/* <Image
          src={logo}
          alt="Company Logo"
          width={50}
          height={50}
          className="mb-4"
        /> */}
       
        <h1 className="mt-4 text-2xl font-extrabold tracking-tight mb-4 text-gray-900">
          Anderson Bank & Insurance Customer Support
        </h1>
        <p className='text-3xl mb-4 font-extrabold'><span className='text-5xl'>👩🏻‍🦰</span> Hello, I'm Sam!</p>
        <p className="text-lg text-gray-600">
          Get assistance anytime with Sam,
        </p>
        <p>an AI Voice Agent.</p>
      </div>

      <button
          className="flex items-center px-6 py-3 bg-[#ffb703] hover:bg-[#f8c915] text-black rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition"
          onClick={handleClick}
        >
          <PhoneIcon className="w-5 h-5 mr-2" />
          Call Sam
      </button>

      <div className="flex flex-col items-center space-y-6 mt-8">
        <div className="flex items-center px-6 py-4 bg-white rounded-lg shadow-md">
          {/* <PhoneIcon className="w-6 h-6 text-blue-500 mr-4" /> */}
          <p className="text-gray-700">
            Need help? Sam is here to provide you quick and accurate answers to your inquiries.
          </p>
        </div>

        
      </div>

      <footer className="absolute bottom-4 text-sm text-gray-500">
        © {new Date().getFullYear()} SGV FSO. All rights reserved.
      </footer>
    </div>
  );
}