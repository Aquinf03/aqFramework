/* eslint-disable @next/next/no-img-element */
import { cn } from "@/lib/utils";
import { siteConfig } from "@/lib/config";

type AquinBrandProps = {
  size?: "sm" | "md" | "lg";
  className?: string;
  href?: string;
};

const sizes = {
  sm: { logo: "h-6", text: "text-lg" },
  md: { logo: "h-8", text: "text-xl sm:text-2xl" },
  lg: { logo: "h-8", text: "text-xl" },
};

export function AquinBrand({ size = "sm", className, href = siteConfig.links.mainSite }: AquinBrandProps) {
  const s = sizes[size];
  return (
    <a href={href} className={cn("inline-flex items-center gap-2.5", className)}>
      <img src="/mainlogo2.png" alt="Aquin" className={s.logo} />
      <span
        className={cn(
          "font-semibold tracking-tighter text-stone-900 font-host-grotesk",
          s.text,
        )}
      >
        Aquin Labs
      </span>
    </a>
  );
}
