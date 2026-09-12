import type { MetadataRoute } from "next";
import { siteConfig } from "@/lib/config";
import { docsSitemapEntries } from "@/lib/docs/metadata";

export default function sitemap(): MetadataRoute.Sitemap {
  const base = siteConfig.url;
  return docsSitemapEntries(base);
}
