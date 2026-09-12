document.querySelectorAll(".copy-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const block = btn.closest(".codeblock");
    const lines = [...block.querySelectorAll(".line")].map((line) => {
      const spans = [...line.querySelectorAll(":scope > span")];
      const textSpan = spans.length > 1 ? spans[1] : spans[0];
      return (textSpan?.textContent || "").replace(/\u00a0/g, " ");
    });
    const text = lines.join("\n").replace(/^\n+|\n+$/g, "");
    try {
      await navigator.clipboard.writeText(text);
      const prev = btn.textContent;
      btn.textContent = "Copied";
      setTimeout(() => { btn.textContent = prev; }, 1400);
    } catch {}
  });
});
