# DIP 数字员工

## 安装依赖

```bash
pnpm install
```

## 启动

```bash
pnpm run dev
```

默认访问地址：[http://localhost:3001](http://localhost:3001)

如需修改本地监听地址，可在 `.env.local` 中设置：

```bash
DEV_HOST=127.0.0.1
DEV_PORT=3001
```

## 调试

### 修改配置

复制 `.env.example` → `.env.local`，修改配置

```bash
DEBUG_ORIGIN=https://your-backend-origin # DIP Studio 服务的访问地址（本地通常是 http://127.0.0.1:3000）
PUBLIC_TOKEN=your_access_token # 可以为空
PUBLIC_REFRESH_TOKEN=your_refresh_token # 可以为空
```

### 跳过认证

在 `.env.local` 中新增配置：

`PUBLIC_SKIP_AUTH=true`

### 切换 admin / 普通用户

在 `.env.local` 中新增配置：

`PUBLIC_IS_ADMIN=true`

### 微应用开发联调（本地入口覆盖）

`apps/dip` 已内置“开发环境微应用入口覆盖”能力，联调时不需要修改源码里的菜单配置（如 `menus.ts`）。

#### 适用场景

- 线上/测试环境默认入口可用，但你需要把某个微应用切换到本地 dev server 调试
- 希望临时切换入口，调试完可快速回退

#### 使用方式

在浏览器控制台执行：

```javascript
localStorage.setItem('DIP_HUB_LOCAL_DEV_MICRO_APPS', JSON.stringify({
  'data-connect': 'http://localhost:8081',
  'my-agent-list': 'http://localhost:8082'
}))
```

然后刷新页面。

说明：

- `key` 必须是微应用 `micro_app.name`（通常与菜单里的 `page.app.name` 一致）
- `value` 是该微应用本地开发服务入口（通常为 `http://localhost:<port>`）

#### 常见问题

- 不生效：确认已正确写入浏览器当前域名的 Local Storage，并在设置后刷新页面
- 名称不匹配：确认 key 与微应用 `micro_app.name` 完全一致
- 跨域问题：确认本地微应用 dev server 已允许主应用域名访问（CORS）

注意：

- 当前实现会读取 `DIP_HUB_LOCAL_DEV_MICRO_APPS` 并覆盖入口地址，未额外限制运行环境
- 建议仅在本地联调时使用，调试完成后及时清理该配置

#### 清理联调配置

清空全部覆盖：

```javascript
localStorage.removeItem('DIP_HUB_LOCAL_DEV_MICRO_APPS')
```

删除某一个微应用覆盖：

```javascript
const config = JSON.parse(localStorage.getItem('DIP_HUB_LOCAL_DEV_MICRO_APPS') || '{}')
delete config['data-connect']
localStorage.setItem('DIP_HUB_LOCAL_DEV_MICRO_APPS', JSON.stringify(config))
```

## 开发质量检查

建议在提交代码前执行：

```bash
pnpm run gate:local
```

`gate:local` 当前包含：

- `pnpm run lint`
- `pnpm run typecheck`

如需一键格式化并修复样式问题，可执行：

```bash
pnpm run check:all
```

## 生产构建

构建：

```bash
pnpm run build
```

本地预览生产版本：

```bash
pnpm run preview
```

## 常见问题

### 1) 端口被占用

默认开发端口是 `3001`。若启动失败，请先释放端口或调整本地运行环境后重试。

### 2) 接口请求失败或代理不生效

请检查 `.env.local` 中 `DEBUG_ORIGIN` 是否配置正确，并确认目标服务可访问。

### 3) 登录状态异常

如本地调试需要，可在 `.env.local` 中启用：

- `PUBLIC_SKIP_AUTH=true`（跳过认证）
- `PUBLIC_IS_ADMIN=true`（管理员视角）
