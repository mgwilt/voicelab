# Connect another device

Use this guide for local network access. The default server accepts connections only from the same computer.

## Prepare HTTPS

Browser microphones and AudioWorklets need a secure context. HTTP on a LAN address does not meet this requirement.

1. Create a development certificate for the server's LAN address or hostname. You can use mkcert for this task.
2. Store the certificate files under ignored `.local/`.
3. Trust the certificate authority (CA) on each test device.
4. Add the server settings below to `.local/server.toml`.

```toml
host = "0.0.0.0"
port = 8443
certfile = ".local/certs/server.pem"
keyfile = ".local/certs/server-key.pem"
ca_file = ".local/certs/rootCA.pem" # optional public CA download
```

Paths are relative to the repository root. The certificate must cover the address that the client uses.
Create a new certificate if that address changes.

## Connect

1. Start the server with `./voicelab`.
2. Open `https://<inference-host>:8443` on a device on the same network.
3. Allow microphone access if the run needs it.

No router forwarding or public deployment is required.
The lab has no account system. Other devices on the LAN can access a server that listens on the network.
The live server does not save browser microphone recordings or transcripts.

## Keep certificates local

Do not share CA private keys or server private keys.
The optional `/local-ca.cer` route serves only the configured public CA certificate.
The application does not change device trust settings.
Keep addresses, certificates, and personal preferences in ignored `.local/` files.

Sources: [Apple certificate trust](https://support.apple.com/en-us/102390),
[MDN microphone requirements](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).
