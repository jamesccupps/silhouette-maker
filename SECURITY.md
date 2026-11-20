# Security policy

Silhouette Maker handles photos of children, so reports are taken seriously.

## Reporting a vulnerability

Please don't open a public issue. Use GitHub's private vulnerability reporting:
**Security → Report a vulnerability** on this repository. Only the maintainer
sees it. Include what an attacker could do and from where (another web site,
your network), the steps to reproduce, and the version (`__version__` in
`app.py`, or the commit).

You should hear back within a week.

## What counts

- **Anything that gets a photo, a silhouette or a name off the computer.** The
  app makes no network requests with your photos. The only download is the
  cut-out model files, fetched once by the `rembg` library.
- **Another web site or another computer using the app.** It listens on
  `127.0.0.1` only and answers only requests addressed to its own address and
  coming from its own page. A page that gets past that check, by DNS rebinding
  or otherwise, is in scope.
- **Reading or writing files outside the app's `data` folder** through the
  page or its API.

## What doesn't

- **Other people who use the same Windows account**, or anyone with access to
  the computer's files: the photos in `data/` are ordinary files and are not
  encrypted.
- **The cut-out models and libraries themselves.** Report those to their own
  projects (see *Credits* in the README).
