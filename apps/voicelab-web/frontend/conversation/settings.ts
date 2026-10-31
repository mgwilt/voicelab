/** Hide direct-run inputs whose values are supplied by the live conversation. */
export function hideFields(root: HTMLElement, titles: readonly string[]): void {
  for (const label of root.querySelectorAll("label.field")) {
    const title = label.querySelector(".field-label")?.textContent;
    if (label instanceof HTMLElement && title !== null && title !== undefined && titles.includes(title)) label.hidden = true;
  }
}
