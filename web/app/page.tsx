import type { Metadata } from "next";
import AuthPortal from "@/components/AuthPortal";
import { siteConfig } from "@/lib/config";
import { constructMetadata } from "@/lib/utils";

export const metadata: Metadata = constructMetadata({
  title: `${siteConfig.name} | Sign in`,
  description: siteConfig.description,
});

export default function HomePage() {
  return <AuthPortal />;
}
