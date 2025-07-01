"use client";
//import Image from 'next/image';
import { ArrowRightCircleIcon, PhoneIcon } from "@heroicons/react/24/solid";
import { useRouter } from "next/navigation";
import Modal from "../../components/Modal";
import { useState } from "react";
import { signInUser, signUpUser } from "../../queries/supabaseQueries";
import Notification from "../../components/Notification";
//import logo from '@/app/assets/logo_ai.png';

export default function Home() {
  const router = useRouter();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isSignUp, setIsSignUp] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const [firstname, setFirstname] = useState("");
  const [lastname, setLastname] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const [notification, setNotification] = useState({ message: "", type: "" });

  const handleClick = () => {
    router.push("/pages/callpage");
  };

  const handleNotificationClose = () => {
    setNotification({ message: "", type: "" });
  };

  const resetFields = () => {
    setFirstname("");
    setLastname("");
    setEmail("");
    setPassword("");
    setConfirmPassword("");
  };

  const handleSubmit = async () => {
    if (isSignUp) {
      if (password !== confirmPassword) {
        setNotification({ message: `Passwords do not match`, type: "error" });
        return;
      }

      const result = await signUpUser({ firstname, lastname, email, password });
      if (result.success) {
        setNotification({
          message: `Account created! You can now sign in.`,
          type: "success",
        });
        setIsSignUp(false);
        resetFields();
      } else {
        setNotification({ message: `${result.error.message}!`, type: "error" });
      }
    } else {
      const result = await signInUser({ email, password });
      if (result.success) {
        setNotification({
          message: `Welcome, ${result.data.firstname}!`,
          type: "success",
        });
        setIsModalOpen(false);
        resetFields();
        setIsLoggedIn(true);
      } else {
        setNotification({ message: `${result.error}!`, type: "error" });
      }
    }
  };

  return (
    <div className="flex p-4 flex-col items-center justify-center min-h-screen bg-gradient-to-b from-blue-50 to-blue-100 text-gray-800">
      <Notification
        message={notification.message}
        type={notification.type}
        onClose={handleNotificationClose}
      />
      <div className="text-center mb-8 flex flex-col items-center">
        {/* <Image
          src={logo}
          alt="Company Logo"
          width={50}
          height={50}
          className="mb-4"
        /> */}

        <h1 className="mt-4 text-2xl font-extrabold tracking-tight mb-4 text-gray-900">
          <span className="text-5xl">👩🏻‍🦰</span> Hello, I'm Sam!
        </h1>
        {/* <p className='text-3xl mb-4 font-extrabold'></p> */}
        <p className="text-lg text-gray-600">Your SGV FSO AI Voice Agent</p>
      </div>
      <div className="bg-white p-6 rounded-2xl shadow-md">
        <h3 className="text-xl font-semibold mb-4 text-gray-800">Features:</h3>
        <ul className="space-y-3 text-gray-700">
          <li>
            📂 <strong>Domain Knowledge:</strong> I can deliver accurate and
            tailored responses based on given data sources.
          </li>
          <li>
            🌤️ <strong>Weather:</strong> Ask me for the current weather and
            time—always up-to-date.
          </li>
          <li>
            ⏱️ <strong>Tools:</strong> I can store and extract data in a
            database, and generate a report for you..
          </li>
        </ul>
      </div>

      <button
        className="flex items-center mt-6 px-6 py-3 bg-[#ffb703] hover:bg-[#f8c915] text-black rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition"
        onClick={handleClick}
      >
        <PhoneIcon className="w-5 h-5 mr-2" />
        Activate Sam
      </button>

      <div className="flex flex-col items-center space-y-6 mt-8">
        <div className="flex items-center px-6 py-4 bg-white rounded-lg shadow-md">
          {/* <PhoneIcon className="w-6 h-6 text-blue-500 mr-4" /> */}
          <p className="text-gray-700">
            Need help? Sam is here to provide you quick and accurate answers to
            your inquiries.
          </p>
        </div>
      </div>

      <footer className="absolute bottom-4 text-sm text-gray-500">
        © {new Date().getFullYear()} SGV FSO. All rights reserved.
      </footer>
    </div>
  );
}
