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
        <p className="text-lg text-gray-600">
          Get assistance anytime with Sam,
        </p>
        <p>an AI Voice Agent.</p>
      </div>

      {/* Conditionally render the button based on login status */}
      {!isLoggedIn ? (
        <button
          className="flex items-center px-6 py-3 bg-[#ffb703] hover:bg-[#f8c915] text-black rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition"
          onClick={() => {
            setIsModalOpen(true);
            setIsSignUp(false);
            resetFields();
          }}
        >
          <ArrowRightCircleIcon className="w-5 h-5 mr-2" />
          Sign In
        </button>
      ) : (
        <button
          className="flex items-center px-6 py-3 bg-[#ffb703] hover:bg-[#f8c915] text-black rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition"
          onClick={handleClick}
        >
          <PhoneIcon className="w-5 h-5 mr-2" />
          Call Sam
        </button>
      )}

      <Modal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)}>
        <h2 className="text-xl font-bold mb-4">
          {isSignUp ? "Sign Up" : "Sign In"}
        </h2>

        {isSignUp && (
          <>
            <input
              type="text"
              placeholder="First Name"
              className="w-full mb-3 p-2 border border-gray-300 rounded"
              value={firstname}
              onChange={(e) => setFirstname(e.target.value)}
            />
            <input
              type="text"
              placeholder="Last Name"
              className="w-full mb-3 p-2 border border-gray-300 rounded"
              value={lastname}
              onChange={(e) => setLastname(e.target.value)}
            />
          </>
        )}

        <input
          type="email"
          placeholder="Email"
          className="w-full mb-3 p-2 border border-gray-300 rounded"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          type="password"
          placeholder="Password"
          className="w-full mb-3 p-2 border border-gray-300 rounded"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        {isSignUp && (
          <input
            type="password"
            placeholder="Confirm Password"
            className="w-full mb-3 p-2 border border-gray-300 rounded"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
          />
        )}

        {!isSignUp ? (
          <p className="text-sm mt-2">
            Don't have an account?{" "}
            <button
              className="text-blue-500 hover:underline"
              onClick={() => setIsSignUp(true)}
            >
              Sign Up
            </button>
          </p>
        ) : (
          <p className="text-sm mt-2">
            Already have an account?{" "}
            <button
              className="text-blue-500 hover:underline"
              onClick={() => setIsSignUp(false)}
            >
              Sign In
            </button>
          </p>
        )}

        <button
          className="mt-4 px-4 py-2  bg-[#ffb703] hover:bg-[#f8c915] text-black  w-full"
          onClick={handleSubmit}
        >
          {isSignUp ? "Register" : "Sign In"}
        </button>
      </Modal>

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
