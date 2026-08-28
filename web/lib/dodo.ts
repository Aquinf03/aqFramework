import DodoPayments from "dodopayments";

let _dodo: DodoPayments | null = null;

const isLive = () => process.env.DODO_ENV === "live";

export function getDodo(): DodoPayments {
  if (_dodo) return _dodo;
  if (!process.env.DODO_API_KEY) throw new Error("DODO_API_KEY is missing");
  _dodo = new DodoPayments({
    bearerToken: process.env.DODO_API_KEY,
    environment: isLive() ? "live_mode" : "test_mode",
  });
  return _dodo;
}

/** @deprecated use getDodo() */
export const dodo = new Proxy({} as DodoPayments, {
  get(_target, prop) {
    return Reflect.get(getDodo(), prop);
  },
});

export function getDodoBase(): string {
  return isLive() ? "https://live.dodopayments.com" : "https://test.dodopayments.com";
}

export const DODO_BASE = getDodoBase();

// 1 credit = 1 training minute
// $0.10 per credit  →  min 5 credits ($0.50), max 1000 credits ($100)
export const CREDIT_RATE_USD = 0.10;
export const CREDIT_MIN = 5;
export const CREDIT_MAX = 1000;

export function creditsToUsd(credits: number): number {
  return parseFloat((credits * CREDIT_RATE_USD).toFixed(2));
}
