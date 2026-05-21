import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

const { navigateMock, locationState, userInfoState, pinnedState, workPlanState, historyState } =
  vi.hoisted(() => ({
    navigateMock: vi.fn(),
    locationState: { pathname: '/' },
    userInfoState: { modules: ['studio', 'store'] as string[] },
    pinnedState: {
      pinnedDigitalHumans: [] as Array<{ id: string; name: string }>,
      fetchSidebarPinnedDigitalHumans: vi.fn(async () => {}),
    },
    workPlanState: {
      plans: [] as any[],
      total: 0,
      fetchPlans: vi.fn(async () => {}),
      refreshPlansOnFocus: vi.fn(async () => {}),
      pausePlan: vi.fn(async () => true),
      resumePlan: vi.fn(async () => true),
      deletePlan: vi.fn(async () => true),
      selectedPlanId: undefined as string | undefined,
      setSelectedPlanId: vi.fn(),
    },
    historyState: {
      sessions: [] as any[],
      total: 0,
      fetchSessions: vi.fn(async () => {}),
      refreshSessionsOnFocus: vi.fn(async () => {}),
      selectedSessionKey: undefined as string | undefined,
      setSelectedSessionKey: vi.fn(),
      deleteHistorySession: vi.fn(async () => true),
    },
  }))

vi.mock('react-router-dom', () => ({
  useNavigate: () => navigateMock,
  useLocation: () => locationState,
}))

vi.mock('antd', () => ({
  message: { useMessage: () => [{}, <div key="msg-holder" />] },
  Modal: { useModal: () => [{ confirm: vi.fn() }, <div key="modal-holder" />] },
}))

vi.mock('@/stores/userInfoStore', () => ({
  useUserInfoStore: (selector: (s: { modules: string[] }) => unknown) => selector(userInfoState),
}))
vi.mock('@/stores/userWorkPlanStore', () => ({
  useUserWorkPlanStore: () => workPlanState,
}))
vi.mock('@/stores/userHistoryStore', () => ({
  useUserHistoryStore: () => historyState,
}))
vi.mock('@/stores/pinnedDigitalHumansStore', () => ({
  usePinnedDigitalHumansStore: (selector: (s: typeof pinnedState) => unknown) =>
    selector(pinnedState),
}))
vi.mock('@/stores/languageStore', () => ({
  useLanguageStore: () => ({ language: 'zh-CN' }),
}))
vi.mock('@/stores/oemConfigStore', () => ({
  useOEMConfigStore: () => ({ getOEMResourceConfig: () => ({ 'logo.png': '/logo.png' }) }),
}))
vi.mock('@/routes/utils', () => ({
  getRouteByPath: () => ({ key: 'home' }),
}))

vi.mock('../../components/StudioMenuSection', () => ({
  StudioMenuSection: () => <div data-testid="studio-menu" />,
}))
vi.mock('../../components/StoreMenuSection', () => ({
  StoreMenuSection: () => <div data-testid="store-menu" />,
}))
vi.mock('../../components/PinnedDigitalHumansSection', () => ({
  PinnedDigitalHumansSection: ({ items }: { items: Array<{ id: string }> }) => (
    <div data-testid="pinned-digital-humans">{items.map((item) => item.id).join(',')}</div>
  ),
}))
vi.mock('../../components/ExternalLinksMenu', () => ({
  ExternalLinksSection: () => <div data-testid="external-links" />,
}))
vi.mock('../../components/SiderFooterUser', () => ({
  SiderFooterUser: () => <div data-testid="footer-user" />,
}))

import AdminSider from '../index'

describe('Sider/AdminSider', () => {
  it('按模块渲染 Studio/Store 区域', () => {
    userInfoState.modules = ['studio', 'store']
    pinnedState.pinnedDigitalHumans = [{ id: 'dh-1', name: 'A' }]
    render(<AdminSider collapsed={false} onCollapse={vi.fn()} layout="entry" />)
    expect(screen.getByTestId('studio-menu')).toBeInTheDocument()
    expect(screen.getByTestId('pinned-digital-humans')).toHaveTextContent('dh-1')
    expect(screen.getByTestId('store-menu')).toBeInTheDocument()
    expect(screen.getByTestId('external-links')).toBeInTheDocument()
    expect(screen.getByTestId('footer-user')).toBeInTheDocument()
    expect(pinnedState.fetchSidebarPinnedDigitalHumans).toHaveBeenCalled()
  })

  it('仅 store 模块时不渲染 Studio 菜单', () => {
    userInfoState.modules = ['store']
    render(<AdminSider collapsed={false} onCollapse={vi.fn()} layout="entry" />)
    expect(screen.queryByTestId('studio-menu')).not.toBeInTheDocument()
    expect(screen.queryByTestId('pinned-digital-humans')).not.toBeInTheDocument()
    expect(screen.getByTestId('store-menu')).toBeInTheDocument()
  })
})
