"use strict";
// The print page: any choice redraws the pages in place; Print opens the browser's print window.
const form = document.getElementById("opts");
let latest = 0;
form.addEventListener("change", async () => {
  const url = "/print?" + new URLSearchParams(new FormData(form));
  const mine = ++latest;
  let page;
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error();
    page = new DOMParser().parseFromString(await r.text(), "text/html");
  } catch { location.href = url; return; } // the app is not answering: let the browser say so
  if (mine !== latest) return; // a newer choice is on its way
  for (const what of ["style", "#sheets", ".count", ".how"]) // the paper size lives in the style
    document.querySelector(what).replaceWith(document.adoptNode(page.querySelector(what)));
  history.replaceState(null, "", url);
});
form.addEventListener("submit", (e) => e.preventDefault()); // Enter in Copies: the change above has done it
document.getElementById("print").addEventListener("click", () => window.print());
