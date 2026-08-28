import { NextRequest, NextResponse } from "next/server";
import { resolveApiKeyDetailed, unauthorizedKeyResponse } from "@/lib/apiKeyAuth";
import { supabaseService } from "@/lib/supabase/service";

export async function GET(req: NextRequest) {
  const auth = await resolveApiKeyDetailed(req.headers.get("authorization"));
  if (auth.status !== "ok") return unauthorizedKeyResponse(auth);

  const { data: profile } = await supabaseService
    .from("profiles")
    .select("email, name, avatar_url")
    .eq("id", auth.userId)
    .single();

  return NextResponse.json({
    user_id: auth.userId,
    email: profile?.email ?? null,
    name: profile?.name ?? null,
    avatar_url: profile?.avatar_url ?? null,
  });
}
