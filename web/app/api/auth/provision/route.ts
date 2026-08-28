// app/api/auth/provision/route.ts

import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";
import { dodo } from "@/lib/dodo";

export async function POST(req: NextRequest) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let body: any = {};
  try {
    body = await req.json();
  } catch (e) {
    // non-JSON body, proceed with empty body
  }

  const billing = body?.billing ?? null;

  const { data: profile, error: profileError } = await supabase
    .from("profiles")
    .select("dodo_customer_id, dodo_subscription_id, email, name")
    .eq("id", user.id)
    .single();

  if (profileError) {
    console.error("[provision] profile fetch error:", profileError.message);
  }

  if (profile?.dodo_subscription_id) {
    try {
      const existing = (await dodo.subscriptions.retrieve(
        profile.dodo_subscription_id,
      )) as any;

      if (existing.status === "active") {
        return NextResponse.json({
          customer_id: profile.dodo_customer_id,
          subscription_id: profile.dodo_subscription_id,
          payment_link: null,
        });
      }

      await supabase
        .from("profiles")
        .update({ dodo_subscription_id: null })
        .eq("id", user.id);
    } catch (e) {
      // Dodo returned 404 or error — ID is stale, clear it
      await supabase
        .from("profiles")
        .update({ dodo_subscription_id: null })
        .eq("id", user.id);
    }
  }

  if (!billing) {
    return NextResponse.json(
      { error: "Billing address required" },
      { status: 400 },
    );
  }

  let customerId = profile?.dodo_customer_id ?? null;

  if (!customerId) {
    try {
      const customer = await dodo.customers.create({
        email: profile?.email ?? user.email!,
        name: billing.name ?? profile?.name ?? user.email!,
      });
      customerId = customer.customer_id;

      await supabase
        .from("profiles")
        .update({ dodo_customer_id: customerId })
        .eq("id", user.id);
    } catch (e) {
      console.error("[provision] customer create FAILED");
      return NextResponse.json(
        { error: "Failed to create customer" },
        { status: 500 },
      );
    }
  }

  try {
    const subscription = await dodo.subscriptions.create({
      customer: { customer_id: customerId },
      product_id: process.env.DODO_PRODUCT_ID!,
      quantity: 1,
      payment_link: true,
      return_url: `${process.env.NEXT_PUBLIC_APP_URL}/login?subscription=activated`,
      billing: {
        city: billing.city,
        state: billing.state,
        street: billing.street,
        zipcode: billing.zipcode,
        country: billing.country,
      },
    });

    await supabase
      .from("profiles")
      .update({ dodo_subscription_id: subscription.subscription_id })
      .eq("id", user.id);

    return NextResponse.json({
      customer_id: customerId,
      subscription_id: subscription.subscription_id,
      payment_link: subscription.payment_link,
    });
  } catch (e: any) {
    console.error("[provision] subscription create FAILED:", e?.message);
    return NextResponse.json(
      { error: e?.message ?? "Failed to create subscription" },
      { status: 500 },
    );
  }
}
