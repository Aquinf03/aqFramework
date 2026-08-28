/* eslint-disable @next/next/no-img-element */
"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { Check, CircleNotch, Copy, Eye, EyeSlash } from "@phosphor-icons/react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import CliTokenSection from "@/components/account/CliTokenSection";
import { PipInstallPill } from "@/components/PipInstallPill";
import ProfileChip from "@/components/account/ProfileChip";
import { siteConfig } from "@/lib/config";

type AuthStep = "email" | "password" | "signup" | "signup-password" | "ready" | "desktop";
type DesktopPhase = "minting" | "ready" | "error";

const ORG_TYPES = [
  "ML Engineer / AI Researcher",
  "Startup or Frontier Lab",
  "University or Research Group",
  "Data Collection / Curation Company",
  "Compliance Vendor or AI Auditor",
  "Consulting Firm",
  "Other",
] as const;

const inputCls =
  "w-full px-4 py-3 rounded-xl border border-stone-200 bg-white text-stone-900 placeholder-stone-400 text-sm focus:outline-none focus:ring-2 focus:ring-stone-300 focus:border-transparent transition-all";

const labelCls = "block text-xs font-mono uppercase tracking-widest text-stone-500 mb-2";

function codeFromDeepLink(link: string): string | null {
  try {
    return new URL(link).searchParams.get("code");
  } catch {
    return null;
  }
}

function AuthPortalInner() {
  const { user, loading: authLoading } = useAuth();
  const supabase = createClient();
  const router = useRouter();
  const searchParams = useSearchParams();
  const viewDesktop = searchParams.get("view") === "desktop";
  const client = (searchParams.get("client") || "").toLowerCase();
  const isCli = client === "cli" || client === "";

  const [step, setStep] = useState<AuthStep>("email");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [orgType, setOrgType] = useState("");
  const [orgName, setOrgName] = useState("");
  const [useCase, setUseCase] = useState("");
  const [twitter, setTwitter] = useState("");
  const [linkedin, setLinkedin] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: "error" | "success"; text: string } | null>(null);

  const [desktopPhase, setDesktopPhase] = useState<DesktopPhase>("minting");
  const [desktopError, setDesktopError] = useState<string | null>(null);
  const [deepLink, setDeepLink] = useState<string | null>(null);
  const [code, setCode] = useState<string | null>(null);
  const [copied, setCopied] = useState<"code" | "link" | null>(null);
  const minted = useRef(false);

  const desktopQuery = useMemo(() => {
    const q = new URLSearchParams();
    q.set("view", "desktop");
    if (client) q.set("client", client);
    return `/?${q.toString()}`;
  }, [client]);

  useEffect(() => {
    if (authLoading) return;
    if (!user) {
      if (viewDesktop) setStep("email");
      return;
    }
    if (user.email) setEmail(user.email);
    if (viewDesktop) {
      setStep("desktop");
      return;
    }
    setStep("ready");
  }, [user, authLoading, viewDesktop]);

  const mintDesktopCode = useCallback(async () => {
    setDesktopPhase("minting");
    setDesktopError(null);
    setCopied(null);
    try {
      const res = await fetch("/api/auth/desktop/mint", { method: "POST" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(
          typeof data.error === "string" ? data.error : "Could not prepare sign-in code.",
        );
      }
      const link = String(data.deep_link || "");
      const raw =
        typeof data.code === "string" && data.code ? data.code : codeFromDeepLink(link);
      if (!raw) throw new Error("Invalid desktop handoff.");
      setCode(raw);
      setDeepLink(link.startsWith("aquin://") ? link : `aquin://auth?code=${encodeURIComponent(raw)}`);
      setDesktopPhase("ready");
    } catch (e) {
      setDesktopError(e instanceof Error ? e.message : "Something went wrong.");
      setDesktopPhase("error");
    }
  }, []);

  useEffect(() => {
    if (step !== "desktop" || !user || minted.current) return;
    minted.current = true;
    void mintDesktopCode();
  }, [step, user, mintDesktopCode]);

  const checkEmailExists = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);
    try {
      const { data, error } = await supabase
        .from("profiles")
        .select("email, is_approved")
        .eq("email", email)
        .maybeSingle();
      if (error) throw error;
      setStep(data ? "password" : "signup");
    } catch (err) {
      setMessage({ type: "error", text: err instanceof Error ? err.message : "An error occurred" });
    } finally {
      setLoading(false);
    }
  };

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);
    try {
      const { data, error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) throw error;
      let { data: profile } = await supabase
        .from("profiles")
        .select("is_approved")
        .eq("id", data.user.id)
        .maybeSingle();
      if (!profile) {
        await supabase.from("profiles").insert({
          id: data.user.id,
          email: data.user.email!,
          is_approved: false,
        });
      }
      if (viewDesktop) {
        router.replace(desktopQuery);
        return;
      }
      setStep("ready");
    } catch (err) {
      setMessage({ type: "error", text: err instanceof Error ? err.message : "An error occurred" });
    } finally {
      setLoading(false);
    }
  };

  const handleSignupNext = (e: React.FormEvent) => {
    e.preventDefault();
    setMessage(null);
    setStep("signup-password");
  };

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: { data: { name: name || null } },
      });
      if (error) throw error;
      if (data.user) {
        const { error: profileError } = await supabase.from("profiles").insert({
          id: data.user.id,
          email: data.user.email!,
          name: name || null,
          avatar_url: null,
          is_approved: false,
          waitlist_role: orgType || null,
          waitlist_use_case: useCase || null,
          waitlist_company: orgName || null,
          waitlist_twitter: twitter || null,
          waitlist_linkedin: linkedin || null,
        });
        if (profileError) throw profileError;

        fetch("/api/waitlist-notify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, email, role: orgType, company: orgName, useCase, twitter, linkedin }),
        }).catch(console.error);

        const { error: signInError } = await supabase.auth.signInWithPassword({ email, password });
        if (signInError) throw signInError;
        if (viewDesktop) {
          router.replace(desktopQuery);
          return;
        }
        setStep("ready");
      }
    } catch (err) {
      setMessage({ type: "error", text: err instanceof Error ? err.message : "An error occurred" });
    } finally {
      setLoading(false);
    }
  };

  const handleForgotPassword = async () => {
    if (!email) {
      setMessage({ type: "error", text: "Please enter your email first" });
      return;
    }
    setLoading(true);
    setMessage(null);
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/auth/reset-password`,
      });
      if (error) throw error;
      setMessage({ type: "success", text: "Password reset link sent to your email." });
    } catch (err) {
      setMessage({ type: "error", text: err instanceof Error ? err.message : "An error occurred" });
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setStep("email");
    setPassword("");
    setName("");
    setOrgType("");
    setOrgName("");
    setUseCase("");
    setTwitter("");
    setLinkedin("");
    setMessage(null);
  };

  const copyText = async (which: "code" | "link", text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(which);
      window.setTimeout(() => setCopied(null), 1600);
    } catch {
      /* select-all still works */
    }
  };

  const openDesktop = () => {
    if (!deepLink) return;
    const a = document.createElement("a");
    a.href = deepLink;
    a.rel = "noopener noreferrer";
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <CircleNotch className="animate-spin h-6 w-6 text-stone-400" weight="bold" />
      </div>
    );
  }

  const showBrandBar = step !== "ready" && step !== "desktop";

  return (
    <div className="h-screen flex flex-col bg-white overflow-hidden">
      {showBrandBar && (
        <div className="flex items-center justify-center px-6 py-4 shrink-0">
          <a href={siteConfig.links.mainSite} className="flex items-center gap-2">
            <img src="/mainlogo2.png" alt="Aquin" className="h-6" />
            <span className="font-semibold tracking-tight text-lg text-stone-900">Aquin Labs</span>
          </a>
        </div>
      )}

      <div className="flex flex-1 items-center justify-center px-4 min-h-0 overflow-y-auto">
        <div className="w-full max-w-sm py-6">
          {step === "ready" && (
            <div className="space-y-6">
              <div className="flex flex-col items-stretch text-center gap-5">
                <div>
                  <a href={siteConfig.links.mainSite} className="inline-flex items-center justify-center gap-2.5 mb-7">
                    <img src="/mainlogo2.png" alt="Aquin" className="h-8" />
                    <span className="font-semibold tracking-tight text-xl sm:text-2xl text-stone-900">
                      Aquin Labs
                    </span>
                  </a>
                  <h2 className="text-2xl font-semibold tracking-tight text-stone-900">You&apos;re signed in</h2>
                  <p className="text-sm text-stone-500 mt-2">
                    Use the desktop app for AI and workspaces. CLI still uses a token below.
                  </p>
                </div>
                <ProfileChip fullWidth />
              </div>

              <Link
                href="/?view=desktop"
                className="w-full flex justify-center items-center gap-2 py-3 px-4 rounded-xl text-sm font-semibold bg-stone-900 text-white hover:bg-stone-800 transition-colors"
              >
                Open Aquin Desktop
              </Link>

              <PipInstallPill variant="hero" size="md" fullWidth />

              <div className="text-left">
                <CliTokenSection />
              </div>
            </div>
          )}

          {step === "desktop" && user && (
            <div className="space-y-6 text-center">
              <a href={siteConfig.links.mainSite} className="inline-flex items-center justify-center gap-2.5">
                <img src="/mainlogo2.png" alt="Aquin" className="h-8" />
                <span className="font-semibold tracking-tight text-xl text-stone-900">Aquin Labs</span>
              </a>

              <div>
                <h1 className="text-2xl font-semibold tracking-tight text-stone-900">
                  {isCli ? "Sign in for Aquin CLI" : "Sign in for CLI / desktop"}
                </h1>
                <p className="text-sm text-stone-500 mt-2">
                  {isCli
                    ? "Copy the code and paste it into the terminal where aq login is waiting."
                    : "Copy the code for the CLI, or open the desktop app with the button below."}
                </p>
              </div>

              <ProfileChip fullWidth />

              {desktopPhase === "minting" && (
                <div className="flex justify-center py-2">
                  <CircleNotch className="animate-spin h-5 w-5 text-stone-400" weight="bold" />
                </div>
              )}

              {desktopError && (
                <p className="text-xs font-mono text-stone-600 border border-stone-200 rounded-xl px-4 py-3 bg-stone-50 text-left">
                  {desktopError}
                </p>
              )}

              {desktopPhase === "ready" && code && (
                <div className="space-y-3 text-left">
                  <p className="text-sm text-stone-500 text-center">
                    In the terminal where <span className="font-mono text-stone-700">aq login</span> is waiting,
                    paste the code or the full <span className="font-mono">aquin://</span> link.
                  </p>
                  <div className="rounded-xl border border-stone-200 bg-stone-50 px-3 py-2.5 space-y-2">
                    <p className="text-[10px] font-medium uppercase tracking-wide text-stone-400">Code</p>
                    <p className="text-[11px] font-mono text-stone-800 break-all select-all leading-relaxed">
                      {code}
                    </p>
                    <button
                      type="button"
                      className="w-full inline-flex items-center justify-center gap-1.5 rounded-lg bg-stone-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-stone-800"
                      onClick={() => void copyText("code", code)}
                    >
                      {copied === "code" ? <Check className="h-3.5 w-3.5" weight="bold" /> : <Copy className="h-3.5 w-3.5" weight="bold" />}
                      {copied === "code" ? "Copied" : "Copy code"}
                    </button>
                  </div>
                  {deepLink && (
                    <div className="rounded-xl border border-stone-200 bg-stone-50 px-3 py-2.5 space-y-2">
                      <p className="text-[10px] font-medium uppercase tracking-wide text-stone-400">aquin:// link</p>
                      <p className="text-[11px] font-mono text-stone-700 break-all select-all">{deepLink}</p>
                      <button
                        type="button"
                        className="w-full inline-flex items-center justify-center gap-1.5 rounded-lg border border-stone-300 px-3 py-1.5 text-xs font-medium text-stone-800 hover:bg-stone-100"
                        onClick={() => void copyText("link", deepLink)}
                      >
                        {copied === "link" ? <Check className="h-3.5 w-3.5" weight="bold" /> : <Copy className="h-3.5 w-3.5" weight="bold" />}
                        {copied === "link" ? "Copied" : "Copy link"}
                      </button>
                    </div>
                  )}
                </div>
              )}

              {desktopPhase === "error" && (
                <button
                  type="button"
                  onClick={() => {
                    minted.current = false;
                    void mintDesktopCode();
                  }}
                  className="w-full py-3 px-4 rounded-xl text-sm font-semibold bg-stone-900 text-white hover:bg-stone-800"
                >
                  Try again
                </button>
              )}

              {desktopPhase === "ready" && deepLink && !isCli && (
                <button
                  type="button"
                  onClick={openDesktop}
                  className="w-full py-3 px-4 rounded-xl text-sm font-semibold bg-stone-900 text-white hover:bg-stone-800"
                >
                  Open Aquin Desktop
                </button>
              )}

              {desktopPhase === "ready" && deepLink && isCli && (
                <button
                  type="button"
                  onClick={openDesktop}
                  className="w-full py-2.5 px-4 rounded-xl text-xs font-medium border border-stone-200 text-stone-600 hover:bg-stone-50"
                >
                  Open Aquin Desktop instead
                </button>
              )}

              <Link href="/" className="block text-xs text-stone-400 hover:text-stone-600">
                &larr; Back to account
              </Link>
            </div>
          )}

          {step === "email" && (
            <div className="space-y-7">
              <div>
                <h1 className="text-3xl font-semibold tracking-tight text-stone-900 leading-tight">
                  {viewDesktop ? "Sign in for Aquin CLI" : "Welcome"}
                </h1>
                <p className="text-sm text-stone-500 mt-2">
                  {viewDesktop
                    ? "Use your aquin.app account, then paste the code into aq login."
                    : "Enter your email to sign in or request access."}
                </p>
              </div>

              <form className="space-y-3.5" onSubmit={checkEmailExists}>
                <div>
                  <label htmlFor="email" className={labelCls}>Email</label>
                  <input
                    id="email"
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className={inputCls}
                    placeholder="you@example.com"
                    autoFocus
                  />
                </div>

                {message && (
                  <p className="text-xs font-mono text-stone-500 border border-stone-200 rounded-xl px-4 py-3 bg-stone-50">
                    {message.text}
                  </p>
                )}

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full flex justify-center items-center gap-2 py-3 px-4 rounded-xl text-sm font-semibold bg-stone-900 text-white hover:bg-stone-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? <CircleNotch className="animate-spin h-4 w-4" weight="bold" /> : null}
                  {loading ? "Checking…" : "Continue"}
                </button>
              </form>
            </div>
          )}

          {step === "password" && (
            <div className="space-y-7">
              <div>
                <p className="text-xs font-mono uppercase tracking-widest text-stone-400 mb-3">Sign in</p>
                <h1 className="text-3xl font-semibold tracking-tight text-stone-900">Enter password</h1>
              </div>

              <form className="space-y-3.5" onSubmit={handleSignIn}>
                <div>
                  <label className={labelCls}>Email</label>
                  <div className="flex items-center gap-2">
                    <input
                      type="email"
                      value={email}
                      disabled
                      className="flex-1 px-4 py-3 rounded-xl border border-stone-200 bg-stone-100 text-stone-500 text-sm"
                    />
                    <button type="button" onClick={resetForm} className="text-xs font-mono text-stone-500 hover:text-stone-900 whitespace-nowrap transition-colors">
                      change
                    </button>
                  </div>
                </div>

                <div>
                  <label htmlFor="password" className={labelCls}>Password</label>
                  <div className="relative">
                    <input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      required
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className={inputCls + " pr-12"}
                      placeholder="••••••••"
                      autoFocus
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute inset-y-0 right-0 pr-4 flex items-center text-stone-400 hover:text-stone-700 transition-colors"
                    >
                      {showPassword ? <EyeSlash className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>

                <div className="flex justify-end">
                  <button type="button" onClick={handleForgotPassword} className="text-xs font-mono text-stone-400 hover:text-stone-700 transition-colors">
                    forgot password?
                  </button>
                </div>

                {message && (
                  <p className={`text-xs font-mono border rounded-xl px-4 py-3 ${
                    message.type === "error"
                      ? "text-stone-500 border-stone-200 bg-stone-50"
                      : "text-stone-600 border-stone-200 bg-[#ffee91]/40"
                  }`}>
                    {message.text}
                  </p>
                )}

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full flex justify-center items-center gap-2 py-3 px-4 rounded-xl text-sm font-semibold bg-stone-900 text-white hover:bg-stone-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? <CircleNotch className="animate-spin h-4 w-4" weight="bold" /> : null}
                  {loading ? "Signing in…" : "Sign In"}
                </button>
              </form>
            </div>
          )}

          {step === "signup" && (
            <div className="space-y-7">
              <div>
                <h1 className="text-3xl font-semibold tracking-tight text-stone-900 leading-tight">Request access</h1>
                <p className="text-sm text-stone-500 mt-1.5 leading-relaxed">
                  Tell us a bit about yourself. We review every request personally.
                </p>
              </div>

              <form className="space-y-3.5" onSubmit={handleSignupNext}>
                <div>
                  <label className={labelCls}>Email</label>
                  <div className="flex items-center gap-2">
                    <input type="email" value={email} disabled className="flex-1 px-4 py-3 rounded-xl border border-stone-200 bg-stone-100 text-stone-500 text-sm" />
                    <button type="button" onClick={resetForm} className="text-xs font-mono text-stone-500 hover:text-stone-900 whitespace-nowrap transition-colors">
                      change
                    </button>
                  </div>
                </div>

                <div>
                  <label htmlFor="name" className={labelCls}>Name</label>
                  <input id="name" type="text" value={name} onChange={(e) => setName(e.target.value)} className={inputCls} placeholder="Jane Smith" autoFocus />
                </div>

                <div>
                  <label className={labelCls}>Organisation</label>
                  <div className="flex gap-2">
                    <Select value={orgType} onValueChange={setOrgType}>
                      <SelectTrigger className="h-11.5 rounded-xl border-stone-200 text-sm text-stone-900 bg-white focus:ring-2 focus:ring-stone-300 shrink-0 w-40">
                        <SelectValue placeholder="Type…" />
                      </SelectTrigger>
                      <SelectContent>
                        {ORG_TYPES.map((t) => (
                          <SelectItem key={t} value={t}>{t}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <input type="text" value={orgName} onChange={(e) => setOrgName(e.target.value)} className={inputCls} placeholder="Name…" />
                  </div>
                </div>

                <div>
                  <label htmlFor="use-case" className={labelCls}>What will you use Aquin for?</label>
                  <textarea id="use-case" value={useCase} onChange={(e) => setUseCase(e.target.value)} rows={2} className={inputCls + " resize-none"} placeholder="Briefly describe your use case…" />
                </div>

                <div>
                  <label htmlFor="twitter" className={labelCls}>X / Twitter URL <span className="normal-case tracking-normal font-sans text-stone-300">(optional)</span></label>
                  <input id="twitter" type="url" value={twitter} onChange={(e) => setTwitter(e.target.value)} className={inputCls} placeholder="https://x.com/yourhandle" />
                </div>

                <div>
                  <label htmlFor="linkedin" className={labelCls}>LinkedIn URL <span className="normal-case tracking-normal font-sans text-stone-300">(optional)</span></label>
                  <input id="linkedin" type="url" value={linkedin} onChange={(e) => setLinkedin(e.target.value)} className={inputCls} placeholder="https://linkedin.com/in/yourprofile" />
                </div>

                <button type="submit" className="w-full flex justify-center items-center gap-2 py-3 px-4 rounded-xl text-sm font-semibold bg-stone-900 text-white hover:bg-stone-800 transition-colors">
                  Continue
                </button>
              </form>
            </div>
          )}

          {step === "signup-password" && (
            <div className="space-y-7">
              <div>
                <p className="text-xs font-mono uppercase tracking-widest text-stone-400 mb-3">Almost there</p>
                <h1 className="text-3xl font-semibold tracking-tight text-stone-900 leading-tight">Secure your spot</h1>
                <p className="text-sm text-stone-500 mt-1.5">Set a password for your account.</p>
              </div>

              <form className="space-y-3.5" onSubmit={handleSignUp}>
                <div>
                  <label htmlFor="signup-password" className={labelCls}>Password</label>
                  <div className="relative">
                    <input
                      id="signup-password"
                      type={showPassword ? "text" : "password"}
                      required
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className={inputCls + " pr-12"}
                      placeholder="••••••••"
                      minLength={6}
                      autoFocus
                    />
                    <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute inset-y-0 right-0 pr-4 flex items-center text-stone-400 hover:text-stone-700 transition-colors">
                      {showPassword ? <EyeSlash className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                  <p className="text-xs font-mono text-stone-400 mt-1.5">min. 6 characters</p>
                </div>

                {message && (
                  <p className={`text-xs font-mono border rounded-xl px-4 py-3 ${
                    message.type === "error"
                      ? "text-stone-500 border-stone-200 bg-stone-50"
                      : "text-stone-600 border-stone-200 bg-[#ffee91]/40"
                  }`}>
                    {message.text}
                  </p>
                )}

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full flex justify-center items-center gap-2 py-3 px-4 rounded-xl text-sm font-semibold bg-stone-900 text-white hover:bg-stone-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? <CircleNotch className="animate-spin h-4 w-4" weight="bold" /> : null}
                  {loading ? "Submitting…" : "Request Access"}
                </button>

                <button type="button" onClick={() => setStep("signup")} className="w-full py-2 text-xs font-mono text-stone-400 hover:text-stone-700 transition-colors">
                  &larr; back
                </button>
              </form>
            </div>
          )}
        </div>
      </div>

      <div className="px-6 py-4 flex flex-wrap items-center justify-center gap-x-4 gap-y-1 shrink-0 text-center">
        <span className="text-xs font-mono text-stone-400">&copy; {new Date().getFullYear()} Aquin Labs</span>
        <a href={`${siteConfig.links.mainSite}/privacy-policy`} className="text-xs font-mono text-stone-400 hover:text-stone-700 transition-colors">Privacy</a>
        <a href={`${siteConfig.links.mainSite}/terms`} className="text-xs font-mono text-stone-400 hover:text-stone-700 transition-colors">Terms</a>
      </div>
    </div>
  );
}

export default function AuthPortal() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center">
          <CircleNotch className="animate-spin h-6 w-6 text-stone-400" weight="bold" />
        </div>
      }
    >
      <AuthPortalInner />
    </Suspense>
  );
}
