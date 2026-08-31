import { kernelStep } from "../core/python.js"

export async function train(argv: string[]): Promise<void> {
  await kernelStep("train", argv)
}

export async function evalCmd(argv: string[]): Promise<void> {
  await kernelStep("eval", argv)
}

export async function checkpoint(argv: string[]): Promise<void> {
  await kernelStep("checkpoint", argv)
}

export async function serve(argv: string[]): Promise<void> {
  await kernelStep("serve", argv)
}
