// A single page built to static files (adapter-static) that the Python server serves:
// nothing runs on a server, the page fetches everything from /api/* in the browser.
export const prerender = true;
export const ssr = false;
