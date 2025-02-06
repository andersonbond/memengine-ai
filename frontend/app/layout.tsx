import "@livekit/components-styles";
import "./globals.css";
import { Public_Sans } from "next/font/google";

export const metadata = {
  title: "Friday, customer support",
  description: "This is my Next.js application.",
};

const publicSans400 = Public_Sans({
  weight: "400",
  subsets: ["latin"],
});

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`h-full ${publicSans400.className}`}>
      <body className="h-full">{children}</body>
    </html>
  );
}