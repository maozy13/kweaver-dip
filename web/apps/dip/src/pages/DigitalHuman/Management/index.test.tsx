import { fireEvent, render, screen } from '@testing-library/react'
import type { PropsWithChildren, ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'

const {
  navigateMock,
  userInfoState,
  pinnedState,
  listServiceState,
  digitalHumanListSpy,
  menuFactorySpy,
} = vi.hoisted(() => ({
  navigateMock: vi.fn(),
  userInfoState: { isAdmin: true },
  pinnedState: {
    pinnedDigitalHumans: [] as Array<{ id: string }>,
    pinSidebarDigitalHuman: vi.fn(async () => true),
    unpinSidebarDigitalHuman: vi.fn(async () => true),
    isPinned: vi.fn((digitalHumanId: string) =>
      pinnedState.pinnedDigitalHumans.some((item) => item.id === digitalHumanId),
    ),
  },
  listServiceState: {
    items: [{ id: 'dh-1', name: 'Agent 1' }],
    loading: false,
    error: undefined,
    searchValue: '',
    handleSearch: vi.fn(),
    handleRefresh: vi.fn(),
  },
  digitalHumanListSpy: vi.fn(),
  menuFactorySpy: vi.fn(() => []),
}))

vi.mock('react-router-dom', () => ({
  useNavigate: () => navigateMock,
}))

vi.mock('react-intl-universal', () => ({
  default: {
    get: (key: string, params?: Record<string, unknown>) =>
      params?.max ? `${key}:${params.max}` : key,
  },
}))

vi.mock('antd', () => ({
  Button: ({ children, onClick }: PropsWithChildren<{ onClick?: () => void }>) => (
    <button type="button" onClick={onClick}>
      {children}
    </button>
  ),
  Tooltip: ({ children }: PropsWithChildren) => <>{children}</>,
  Spin: () => <div data-testid="spin" />,
  message: { useMessage: () => [{}, <div key="msg-holder" />] },
}))

vi.mock('@/apis', () => ({
  getDigitalHumanList: vi.fn(),
}))

vi.mock('@/stores/userInfoStore', () => ({
  useUserInfoStore: (selector: (s: typeof userInfoState) => unknown) => selector(userInfoState),
}))

vi.mock('@/stores/pinnedDigitalHumansStore', () => ({
  MAX_PINNED_SIDEBAR_DIGITAL_HUMANS: 8,
  usePinnedDigitalHumansStore: (
    selector: (s: typeof pinnedState & { fetchSidebarPinnedDigitalHumans?: () => Promise<void> }) => unknown,
  ) => selector(pinnedState),
}))

vi.mock('@/hooks/useListService', () => ({
  useListService: () => listServiceState,
}))

vi.mock('@/components/DigitalHumanList', () => ({
  default: (props: {
    digitalHumans: Array<{ id: string }>
    cardTrailing?: (
      digitalHuman: { id: string },
      opts: { cardHovered: boolean; actionMenuVisible: boolean }
    ) => ReactNode
    menuItems?: (digitalHuman: { id: string }) => unknown[]
  }) => {
    digitalHumanListSpy(props)
    const item = props.digitalHumans[0]
    const menuItems = props.menuItems?.(item) ?? []
    return (
      <div>
        <div data-testid="digital-human-list">{item.id}</div>
        <div data-testid="card-trailing">
          {props.cardTrailing?.(item, { cardHovered: true, actionMenuVisible: true })}
        </div>
        <div data-testid="card-trailing-idle">
          {props.cardTrailing?.(item, { cardHovered: false, actionMenuVisible: false })}
        </div>
        <div data-testid="card-trailing-menu-open">
          {props.cardTrailing?.(item, { cardHovered: false, actionMenuVisible: true })}
        </div>
        <div data-testid="menu-items-count">{String(menuItems.length)}</div>
        <button
          type="button"
          onClick={() => {
            const pinMenuItem = menuItems.find((entry: any) =>
              ['pinSidebar', 'unpinSidebar'].includes(String(entry?.key ?? '')),
            ) as any
            pinMenuItem?.onClick?.({ domEvent: { stopPropagation: vi.fn() } })
          }}
        >
          trigger-menu-pin
        </button>
      </div>
    )
  },
}))

vi.mock('./utils', () => ({
  getDigitalHumanManagementMenuItems: (...args: unknown[]) => menuFactorySpy(...args),
}))

vi.mock('@/components/DigitalHumanSetting/ActionModal/DeleteModal', () => ({
  default: () => null,
}))
vi.mock('@/components/Empty', () => ({
  default: () => <div data-testid="empty" />,
}))
vi.mock('@/components/IconFont', () => ({
  default: ({ type }: { type: string }) => <span>{type}</span>,
}))
vi.mock('@/components/SearchInput', () => ({
  default: () => <div data-testid="search-input" />,
}))

import Management from './index'

describe('DigitalHuman/Management', () => {
  it('管理员列表通过扩展菜单提供固定入口，悬浮时不直接显示钉按钮', async () => {
    userInfoState.isAdmin = true
    pinnedState.pinnedDigitalHumans = []
    menuFactorySpy.mockImplementation((_digitalHuman, _options, _onClick) => [])
    render(<Management />)

    expect(screen.getByTestId('digital-human-list')).toHaveTextContent('dh-1')
    expect(screen.getByTestId('menu-items-count')).toHaveTextContent('0')
    expect(screen.getByTestId('card-trailing')).toBeEmptyDOMElement()
    expect(menuFactorySpy).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'dh-1' }),
      { isPinnedInSidebar: false },
      expect.any(Function),
    )
  })

  it('管理员从扩展菜单固定数字员工', () => {
    userInfoState.isAdmin = true
    pinnedState.pinnedDigitalHumans = []
    menuFactorySpy.mockImplementation((_digitalHuman, _options, onClick) => [
      {
        key: 'pinSidebar',
        onClick: () => onClick('pinSidebar'),
      },
    ])
    render(<Management />)

    fireEvent.click(screen.getByRole('button', { name: 'trigger-menu-pin' }))
    expect(pinnedState.pinSidebarDigitalHuman).toHaveBeenCalledWith('dh-1')
  })

  it('管理员已固定时仅在未悬浮态显示静态钉图标，并从扩展菜单取消固定', () => {
    userInfoState.isAdmin = true
    pinnedState.pinnedDigitalHumans = [{ id: 'dh-1' }]
    menuFactorySpy.mockImplementation((_digitalHuman, _options, onClick) => [
      {
        key: 'unpinSidebar',
        onClick: () => onClick('unpinSidebar'),
      },
    ])
    render(<Management />)

    expect(screen.getByTestId('card-trailing')).toBeEmptyDOMElement()
    expect(screen.getByTestId('card-trailing-idle')).toHaveTextContent('icon-solid-pin')
    expect(screen.getByTestId('card-trailing-menu-open')).toBeEmptyDOMElement()
    fireEvent.click(screen.getByRole('button', { name: 'trigger-menu-pin' }))
    expect(pinnedState.unpinSidebarDigitalHuman).toHaveBeenCalledWith('dh-1')
  })
})
