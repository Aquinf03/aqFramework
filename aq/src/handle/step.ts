import { kernelStep, runKernel } from "../core/python.js"
import { assertTrain } from "../core/schema.js"
import { shouldAutoPlot } from "../lib/plot-config.js"

export async function train(argv: string[]): Promise<void> {
  await kernelStep("train", argv)
  const trainDir = assertTrain(argv[0] ?? ".")
  if (shouldAutoPlot(trainDir)) {
    try {
      await runKernel(trainDir, { op: "plot", kind: "all" })
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      process.stderr.write(`plot (auto): ${msg}\n`)
    }
  }
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
