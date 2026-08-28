import type { Metadata, Viewport } from "next";
import { Roboto } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/contexts/AuthContext";
import { cn, constructMetadata } from "@/lib/utils";
import { siteConfig } from "@/lib/config";

const roboto = Roboto({
  weight: ["300", "400", "500", "700"],
  subsets: ["latin"],
  variable: "--font-roboto",
});

export const metadata: Metadata = constructMetadata({
  title: `${siteConfig.name} | Account`,
});

export const viewport: Viewport = {
  colorScheme: "light",
  themeColor: [{ media: "(prefers-color-scheme: light)", color: "white" }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={cn(
          `${roboto.variable} min-h-screen bg-background overflow-x-hidden antialiased w-full mx-auto scroll-smooth font-sans`,
        )}
      >
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
