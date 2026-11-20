"use strict";
// The print page: any choice redraws the preview; Print opens the browser's print window.
const form = document.getElementById("opts");
form.addEventListener("change", () => form.submit());
document.getElementById("print").addEventListener("click", () => window.print());
