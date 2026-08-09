/**
 * Serveur HTTPS local + reverse-proxy API/WS vers tars-voice.
 * Nécessaire pour getUserMedia sur téléphone (LAN) : page HTTPS,
 * et pour éviter le mixed-content (API/WS passent par le même origin).
 */
import { createServer } from "node:https";
import { parse } from "node:url";
import next from "next";
import selfsigned from "selfsigned";
import httpProxy from "http-proxy";

const dev = process.env.NODE_ENV !== "production";
const hostname = process.env.HOSTNAME || "0.0.0.0";
const port = Number(process.env.PORT || 3000);
const voice = (process.env.VOICE_INTERNAL_URL || "http://127.0.0.1:9743").replace(
  /\/$/,
  ""
);

const app = next({ dev, hostname, port });
const handle = app.getRequestHandler();

const proxy = httpProxy.createProxyServer({
  target: voice,
  ws: true,
  changeOrigin: true,
  xfwd: true,
});

proxy.on("error", (err, _req, res) => {
  console.error("[proxy]", err.message);
  if (res && !res.headersSent && typeof res.writeHead === "function") {
    res.writeHead(502, { "Content-Type": "text/plain" });
    res.end("Bad gateway (voice API)");
  }
});

function buildCert() {
  const extra = (process.env.HTTPS_SANS || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const altNames = [
    { type: 2, value: "localhost" },
    { type: 7, ip: "127.0.0.1" },
    { type: 7, ip: "0.0.0.0" },
  ];
  for (const s of extra) {
    if (/^\d+\.\d+\.\d+\.\d+$/.test(s)) altNames.push({ type: 7, ip: s });
    else altNames.push({ type: 2, value: s });
  }
  const pems = selfsigned.generate([{ name: "commonName", value: "TARS local" }], {
    days: 365,
    keySize: 2048,
    algorithm: "sha256",
    extensions: [{ name: "subjectAltName", altNames }],
  });
  return { key: pems.private, cert: pems.cert };
}

function shouldProxy(pathname) {
  return (
    pathname === "/ws" ||
    pathname.startsWith("/api/") ||
    pathname === "/health" ||
    pathname.startsWith("/static/")
  );
}

app.prepare().then(() => {
  const { key, cert } = buildCert();
  const server = createServer({ key, cert }, (req, res) => {
    const parsed = parse(req.url || "/", true);
    const pathname = parsed.pathname || "/";
    if (shouldProxy(pathname)) {
      proxy.web(req, res, { target: voice });
      return;
    }
    handle(req, res, parsed);
  });

  server.on("upgrade", (req, socket, head) => {
    const pathname = parse(req.url || "/").pathname || "/";
    if (pathname === "/ws") {
      proxy.ws(req, socket, head, { target: voice });
      return;
    }
    socket.destroy();
  });

  server.listen(port, hostname, () => {
    console.log(`> TARS web HTTPS https://localhost:${port}`);
    console.log(`> Proxy voice → ${voice}`);
    console.log(`> Sur téléphone : https://<IP_PC>:${port} (accepter le certificat)`);
    if (process.env.HTTPS_SANS) {
      console.log(`> SAN cert : ${process.env.HTTPS_SANS}`);
    }
  });
});
