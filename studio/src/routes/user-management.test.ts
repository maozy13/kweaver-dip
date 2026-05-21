import type { NextFunction, Request, Response, Router } from "express";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { UserManagementAdapter } from "../adapters/user-management-adapter";
import type { DigitalEmployeeTokenAdapter } from "../adapters/digital-employee-token-adapter";
import { HttpError } from "../errors/http-error";
import { readOptionalBearerToken, readOptionalForwardToken } from "./proxy-auth";
import { createUserManagementRouter } from "./user-management";

afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

/**
 * Creates a minimal response double with chainable methods.
 *
 * @returns The mocked response object.
 */
function createResponseDouble(): Response {
  const response = {
    status: vi.fn(),
    send: vi.fn(),
    end: vi.fn(),
    setHeader: vi.fn()
  } as unknown as Response;

  vi.mocked(response.status).mockReturnValue(response);

  return response;
}

/**
 * Locates an Express route handler by path and HTTP method.
 *
 * @param router The Express router.
 * @param method HTTP method.
 * @param path Route path string.
 * @returns The handler function, if any.
 */
function findHandler(
  router: Router,
  method: "get" | "post",
  path: string
):
  | ((
      request: Request,
      response: Response,
      next: NextFunction
    ) => Promise<void>)
  | undefined {
  const layer = router.stack.find((item) => {
    const route = item.route;

    if (!route || route.path !== path) {
      return false;
    }

    return Boolean((route.methods as Record<string, boolean>)[method]);
  });

  return layer?.route?.stack[0]?.handle;
}

/**
 * Creates a user-management adapter test double.
 *
 * @returns A mocked adapter object.
 */
function createAdapterDouble(): UserManagementAdapter {
  return {
    listApps: vi.fn(),
    findAppById: vi.fn(),
    createApp: vi.fn(),
    createAppToken: vi.fn()
  };
}

function createTokenAdapterDouble(): DigitalEmployeeTokenAdapter {
  return {
    findAppId: vi.fn(),
    hasStudioAppToken: vi.fn().mockResolvedValue(false),
    findKweaverToken: vi.fn(),
    findBknScope: vi.fn(),
    upsertStudioAppToken: vi.fn(),
    upsertDigitalEmployee: vi.fn(),
    upsertAppId: vi.fn(),
    upsertKweaverToken: vi.fn(),
    upsertBknScope: vi.fn(),
    deleteKweaverToken: vi.fn(),
    markDigitalEmployeeDeleted: vi.fn()
  };
}

describe("createUserManagementRouter", () => {
  const appsPath = "/api/dip-studio/v1/user-management/apps";
  const tokensPath = "/api/dip-studio/v1/user-management/console/app-tokens";

  it("registers all user-management routes", () => {
    const router = createUserManagementRouter(
      createAdapterDouble(),
      createTokenAdapterDouble()
    ) as Router;

    expect(findHandler(router, "get", appsPath)).toBeDefined();
    expect(findHandler(router, "post", appsPath)).toBeDefined();
    expect(findHandler(router, "post", tokensPath)).toBeDefined();
  });

  it("forwards list requests with bearer token", async () => {
    const adapter = createAdapterDouble();
    const tokenAdapter = createTokenAdapterDouble();
    vi.mocked(tokenAdapter.hasStudioAppToken)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce(false);
    vi.mocked(adapter.listApps).mockResolvedValue({
      status: 200,
      headers: new Headers({ "content-type": "application/json" }),
      body: JSON.stringify({
        entries: [
          { id: "app-1", name: "App 1" },
          { id: "app-2", name: "App 2" }
        ],
        total_count: 2
      })
    });
    const handler = findHandler(
      createUserManagementRouter(adapter, tokenAdapter) as Router,
      "get",
      appsPath
    );
    const response = createResponseDouble();
    const next = vi.fn<NextFunction>();

    await handler?.(
      {
        query: { keyword: "agent" },
        headers: { authorization: "Bearer token-1" }
      } as unknown as Request,
      response,
      next
    );

    expect(adapter.listApps).toHaveBeenCalledWith(
      { keyword: "agent" },
      "token-1"
    );
    expect(response.status).toHaveBeenCalledWith(200);
    expect(tokenAdapter.hasStudioAppToken).toHaveBeenNthCalledWith(1, "app-1");
    expect(tokenAdapter.hasStudioAppToken).toHaveBeenNthCalledWith(2, "app-2");
    expect(response.send).toHaveBeenCalledWith(JSON.stringify({
      entries: [
        { id: "app-1", name: "App 1", has_kweaver_token: true },
        { id: "app-2", name: "App 2", has_kweaver_token: false }
      ],
      total_count: 2
    }));
    expect(next).not.toHaveBeenCalled();
  });

  it("forwards create app token requests", async () => {
    const adapter = createAdapterDouble();
    vi.mocked(adapter.createAppToken).mockResolvedValue({
      status: 200,
      headers: new Headers(),
      body: "{\"token\":\"t\"}"
    });
    const handler = findHandler(
      createUserManagementRouter(adapter, createTokenAdapterDouble()) as Router,
      "post",
      tokensPath
    );
    const response = createResponseDouble();
    const next = vi.fn<NextFunction>();

    await handler?.(
      {
        body: { id: "app-1" },
        headers: { authorization: "Bearer token-1" }
      } as unknown as Request,
      response,
      next
    );

    expect(adapter.createAppToken).toHaveBeenCalledWith(
      { id: "app-1" },
      "token-1"
    );
    expect(response.send).toHaveBeenCalledWith("{\"token\":\"t\"}");
  });

  it("creates and stores an app token after creating an app account", async () => {
    const adapter = createAdapterDouble();
    const tokenAdapter = createTokenAdapterDouble();
    vi.mocked(adapter.createApp).mockResolvedValue({
      status: 201,
      headers: new Headers(),
      body: "{\"id\":\"app-1\"}"
    });
    vi.mocked(adapter.createAppToken).mockResolvedValue({
      status: 200,
      headers: new Headers(),
      body: "{\"token\":\"kw-token-1\"}"
    });
    const handler = findHandler(
      createUserManagementRouter(adapter, tokenAdapter) as Router,
      "post",
      appsPath
    );
    const response = createResponseDouble();
    const next = vi.fn<NextFunction>();

    await handler?.(
      {
        body: { name: "App 1", password: "" },
        headers: { authorization: "Bearer token-1" }
      } as unknown as Request,
      response,
      next
    );

    expect(adapter.createApp).toHaveBeenCalledWith(
      { name: "App 1", password: "" },
      "token-1"
    );
    expect(adapter.createAppToken).toHaveBeenCalledWith(
      { id: "app-1" },
      "token-1"
    );
    expect(tokenAdapter.upsertStudioAppToken).toHaveBeenCalledWith(
      "app-1",
      "kw-token-1"
    );
    expect(response.status).toHaveBeenCalledWith(201);
    expect(response.send).toHaveBeenCalledWith("{\"id\":\"app-1\"}");
  });
});

describe("readOptionalBearerToken", () => {
  it("returns undefined when authorization is absent", () => {
    expect(readOptionalBearerToken({ headers: {} } as Request)).toBeUndefined();
  });

  it("returns the bearer token and rejects malformed values", () => {
    expect(
      readOptionalBearerToken({
        headers: { authorization: "Bearer token-1" }
      } as Request)
    ).toBe("token-1");
    expect(() =>
      readOptionalBearerToken({
        headers: { authorization: "Basic token-1" }
      } as Request)
    ).toThrowError(HttpError);
  });
});

describe("readOptionalForwardToken", () => {
  it("returns undefined when authorization is absent", () => {
    expect(readOptionalForwardToken({ headers: {} } as Request)).toBeUndefined();
  });

  it("unwraps bearer tokens and accepts raw token passthrough", () => {
    expect(
      readOptionalForwardToken({
        headers: { authorization: "Bearer token-1" }
      } as Request)
    ).toBe("token-1");
    expect(
      readOptionalForwardToken({
        headers: { authorization: "raw-token-2" }
      } as Request)
    ).toBe("raw-token-2");
  });
});
