/** Typed element construction shared by the small view components. */
export function element<K extends keyof HTMLElementTagNameMap>(
  tag: K, className = "", content = "",
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = content;
  return node;
}

export function labeled<T extends HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>(
  title: string, input: T, hint = "",
): HTMLLabelElement {
  const label = element("label", "field");
  label.append(element("span", "field-label", title), input);
  if (hint !== "") label.append(element("span", "field-hint", hint));
  return label;
}

export function integerInput(value: number, min: number, max: number): HTMLInputElement {
  const input = element("input");
  input.type = "number";
  input.value = String(value);
  input.min = String(min);
  input.max = String(max);
  input.step = "1";
  input.required = true;
  return input;
}

export function readInteger(input: HTMLInputElement): number {
  const value = input.valueAsNumber;
  if (!input.checkValidity() || !Number.isSafeInteger(value)) {
    input.reportValidity();
    throw new Error("Enter a whole number within the displayed range.");
  }
  return value;
}
