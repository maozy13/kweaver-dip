import type { Request } from "express";

import { HttpError } from "../errors/http-error";

/**
 * Reads an optional bearer token from the incoming request for upstream forwarding.
 *
 * @param request Incoming Express request.
 * @returns The bearer token when present and well-formed.
 */
export function readOptionalBearerToken(request: Pick<Request, "headers">): string | undefined {
  const authorization = request.headers.authorization;

  if (typeof authorization !== "string" || authorization.trim().length === 0) {
    return undefined;
  }

  const [scheme, token] = authorization.trim().split(/\s+/, 2);

  if (scheme !== "Bearer" || token === undefined || token.trim().length === 0) {
    throw new HttpError(401, "Authorization header must use Bearer token");
  }

  return token.trim();
}

/**
 * Reads an optional token for downstream forwarding without enforcing the
 * incoming scheme. `Bearer <token>` is unwrapped; any other non-empty header
 * value is treated as the token payload directly.
 *
 * @param request Incoming Express request.
 * @returns A normalized token value when present.
 */
export function readOptionalForwardToken(request: Pick<Request, "headers">): string | undefined {
  const authorization = request.headers.authorization;

  if (typeof authorization !== "string" || authorization.trim().length === 0) {
    return undefined;
  }

  const trimmed = authorization.trim();
  const bearerMatch = /^Bearer\s+(.+)$/i.exec(trimmed);

  if (bearerMatch) {
    const token = bearerMatch[1]?.trim();
    return token && token.length > 0 ? token : undefined;
  }

  return trimmed;
}
