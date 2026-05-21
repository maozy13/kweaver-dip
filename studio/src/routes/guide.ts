import { Router, type NextFunction, type Request, type Response } from "express";

import { HttpError } from "../errors/http-error";
import { DefaultGuideLogic, type GuideLogic } from "../logic/guide";
import type {
  GuideStatusResponse,
  InitializeGuideRequest,
  OpenClawDetectedConfig
} from "../types/guide";

const guideLogic = new DefaultGuideLogic();

/**
 * Builds the guide router.
 *
 * @param logic Optional guide logic override.
 * @returns The router exposing DIP Studio guide endpoints.
 */
export function createGuideRouter(
  logic: GuideLogic = guideLogic
): Router {
  const router = Router();

  router.get(
    "/api/dip-studio/v1/guide/status",
    async (
      _request: Request,
      response: Response<GuideStatusResponse>,
      next: NextFunction
    ): Promise<void> => {
      try {
        response.status(200).json(await logic.getStatus());
      } catch (error) {
        next(error instanceof HttpError ? error : new HttpError(502, "Failed to query guide status"));
      }
    }
  );

  router.get(
    "/api/dip-studio/v1/guide/openclaw-config",
    async (
      request: Request,
      response: Response<OpenClawDetectedConfig>,
      next: NextFunction
    ): Promise<void> => {
      try {
        response.status(200).json(
          await logic.getOpenClawConfig({
            requestHost: readRequestHost(request),
            requestOrigin: readRequestOrigin(request)
          })
        );
      } catch (error) {
        next(
          error instanceof HttpError
            ? error
            : new HttpError(502, "Failed to query OpenClaw config")
        );
      }
    }
  );

  router.post(
    "/api/dip-studio/v1/guide/initialize",
    async (
      request: Request<unknown, unknown, InitializeGuideRequest>,
      response: Response,
      next: NextFunction
    ): Promise<void> => {
      try {
        await logic.initialize(readInitializeGuideRequestBody(request.body));
        response.sendStatus(200);
      } catch (error) {
        next(error instanceof HttpError ? error : new HttpError(502, "Failed to initialize DIP Studio"));
      }
    }
  );

  return router;
}

/**
 * Reads the best available host name from proxy-aware request headers.
 *
 * @param request Express request.
 * @returns The raw host value when available.
 */
export function readRequestHost(request: Request): string | undefined {
  const forwardedHost = request.headers["x-forwarded-host"];

  if (Array.isArray(forwardedHost)) {
    return forwardedHost[0];
  }

  return forwardedHost ?? request.headers.host;
}

/**
 * Reads the best available request origin from proxy-aware request headers.
 *
 * @param request Express request.
 * @returns The HTTP(S) origin used to access this guide endpoint.
 */
export function readRequestOrigin(request: Request): string | undefined {
  const host = readRequestHost(request);

  if (host === undefined) {
    return undefined;
  }

  const forwardedProto = request.headers["x-forwarded-proto"];
  const rawProtocol = Array.isArray(forwardedProto)
    ? forwardedProto[0]
    : forwardedProto;
  const protocol = rawProtocol?.split(",", 1)[0]?.trim() || request.protocol || "http";

  return `${protocol}://${host}`;
}

/**
 * Validates the guide initialization request body.
 *
 * @param requestBody Raw parsed request body.
 * @returns The validated initialization payload.
 * @throws {HttpError} Thrown when the request body is invalid.
 */
export function readInitializeGuideRequestBody(
  requestBody: unknown
): InitializeGuideRequest {
  if (typeof requestBody !== "object" || requestBody === null || Array.isArray(requestBody)) {
    throw new HttpError(400, "Guide initialize request body must be a JSON object");
  }

  const body = requestBody as Partial<InitializeGuideRequest>;

  if (typeof body.openclaw_address !== "string" || body.openclaw_address.trim() === "") {
    throw new HttpError(400, "openclaw_address is required");
  }

  if (typeof body.openclaw_token !== "string" || body.openclaw_token.trim() === "") {
    throw new HttpError(400, "openclaw_token is required");
  }

  const kweaverBaseUrl =
    typeof body.kweaver_base_url === "string" ? body.kweaver_base_url.trim() : "";

  return {
    openclaw_address: body.openclaw_address.trim(),
    openclaw_token: body.openclaw_token.trim(),
    ...(typeof body.kweaver_base_url === "string"
      ? { kweaver_base_url: kweaverBaseUrl }
      : {})
  };
}
