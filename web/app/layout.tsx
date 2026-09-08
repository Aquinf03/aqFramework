import type { Metadata, Viewport } from "next";
import { Roboto, Cardo, Host_Grotesk } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/contexts/AuthContext";
import { cn, constructMetadata } from "@/lib/utils";
import { siteConfig } from "@/lib/config";

const roboto = Roboto({
  weight: ["300", "400", "500", "700"],
  subsets: ["latin"],
  variable: "--font-roboto",
});

const cardo = Cardo({
  weight: ["400", "700"],
  subsets: ["latin"],
  variable: "--font-cardo",
});

/** Ready via `font-host-grotesk` / `var(--font-host-grotesk)`. */
const hostGrotesk = Host_Grotesk({
  subsets: ["latin"],
  variable: "--font-host-grotesk",
});

export const metadata: Metadata = constructMetadata({
  title: `${siteConfig.name} | Account`,
  description: siteConfig.description,
});

export const viewport: Viewport = {
  colorScheme: "light",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f5f5f3" },
  ],
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
          `${roboto.variable} ${cardo.variable} ${hostGrotesk.variable} min-h-screen bg-background overflow-x-hidden antialiased w-full mx-auto scroll-smooth font-host-grotesk`,
        )}
      >
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
