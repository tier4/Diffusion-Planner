export const ui = {
  page: {
    padding: "16px",
    fontFamily: '"Inter", "Segoe UI", "Roboto", sans-serif',
    fontSize: "15px",
    lineHeight: 1.45,
  } as const,
  section: {
    border: "1px solid #d1d5db",
    borderRadius: "12px",
    padding: "12px",
    marginBottom: "12px",
    background: "#ffffff",
  } as const,
  title: {
    fontSize: "20px",
    margin: "0 0 10px 0",
    fontWeight: 700,
  } as const,
  subtitle: {
    fontSize: "16px",
    margin: "0 0 8px 0",
    fontWeight: 650,
  } as const,
  slider: {
    width: "100%",
    marginTop: "6px",
  } as const,
};

export function parseProgress(text: string): {
  sampleText: string;
  fileText: string;
  preferenceText: string;
} {
  const lines = text.split("\n").map((line) => line.trim());
  return {
    sampleText: lines.find((line) => line.startsWith("Sample:")) ?? "-",
    fileText: lines.find((line) => line.startsWith("File:"))?.replace("File:", "").trim() ?? "-",
    preferenceText: lines.find((line) => line.startsWith("Preferences collected:")) ?? "-",
  };
}

export function parseHistory(text: string): string[] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("- "))
    .map((line) => line.slice(2));
}

export function parseMarkdownTable(markdown: string): { headers: string[]; rows: string[][] } {
  const lines = markdown
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("|"));

  if (lines.length < 2) {
    return { headers: [], rows: [] };
  }

  const splitRow = (row: string) =>
    row
      .split("|")
      .map((cell) => cell.trim())
      .filter((cell) => cell.length > 0);

  const headers = splitRow(lines[0]);
  const rows = lines.slice(2).map(splitRow);
  return { headers, rows };
}
