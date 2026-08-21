# Security

## API keys

`configure.sh` writes API keys to `.env.8000` and `.env.8001`. These files are
ignored by Git. Never commit them, paste them into issues, or include them in
terminal screenshots.

If a key is exposed, generate a new one with:

```bash
openssl rand -hex 32
```

Update both environment files and restart the two instances.

## Network exposure

SGLang listens on `0.0.0.0` so that another machine on a trusted network can
connect. The service does not configure TLS. Do not expose ports 8000 or 8001
directly to the public Internet. Use a firewall, VPN, SSH tunnel, or a TLS
reverse proxy with access control.

## Reporting

Please report security issues privately to the repository owner rather than
opening a public issue containing credentials or infrastructure details.
