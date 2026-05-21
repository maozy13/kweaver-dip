import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

const { navigateMock, locationState, pinnedStoreState } = vi.hoisted(() => ({
  navigateMock: vi.fn(),
  locationState: { pathname: '/' },
  pinnedStoreState: {
    unpinSidebarDigitalHuman: vi.fn(async () => true),
  },
}))

vi.mock('react-router-dom', () => ({
  useNavigate: () => navigateMock,
  useLocation: () => locationState,
}))

vi.mock('react-intl-universal', () => ({
  default: {
    get: (key: string, params?: Record<string, unknown>) =>
      params?.count ? `${key}:${params.count}` : key,
  },
}))

vi.mock('antd', () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Popover: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

vi.mock('@/components/AppIcon', () => ({
  default: ({ name }: { name: string }) => <span>{name}</span>,
}))

vi.mock('@/components/IconFont', () => ({
  default: ({ type }: { type: string }) => <span>{type}</span>,
}))

vi.mock('@/utils/digital-human/resolveDigitalHumanIcon', () => ({
  resolveDigitalHumanIconSrc: () => '',
}))

vi.mock('@/stores/pinnedDigitalHumansStore', () => ({
  usePinnedDigitalHumansStore: (
    selector: (state: typeof pinnedStoreState) => unknown,
  ) => selector(pinnedStoreState),
}))

import {
  SIDEBAR_OPEN_DH_SESSION_LOCATION_KEY,
  SIDEBAR_REOPEN_DH_SESSION_LOCATION_KEY,
} from '@/routes/types'
import { PinnedDigitalHumansSection } from '../PinnedDigitalHumansSection'

describe('Sider/PinnedDigitalHumansSection', () => {
  it('点击未激活的固定数字员工时带会话入口 state 跳转', () => {
    locationState.pathname = '/'
    render(<PinnedDigitalHumansSection items={[{ id: 'dh-1', name: 'Agent 1' }]} />)

    fireEvent.click(screen.getAllByText('Agent 1')[1]!.closest('button')!)
    expect(navigateMock).toHaveBeenCalledWith('/studio/digital-human/dh-1', {
      state: {
        [SIDEBAR_OPEN_DH_SESSION_LOCATION_KEY]: true,
      },
    })
  })

  it('点击当前激活项时带 reopen state 重新打开会话', () => {
    locationState.pathname = '/studio/digital-human/dh-1'
    render(<PinnedDigitalHumansSection items={[{ id: 'dh-1', name: 'Agent 1' }]} />)

    fireEvent.click(screen.getAllByText('Agent 1')[1]!.closest('button')!)
    expect(navigateMock).toHaveBeenCalledWith(
      '/studio/digital-human/dh-1',
      expect.objectContaining({
        replace: true,
        state: expect.objectContaining({
          [SIDEBAR_OPEN_DH_SESSION_LOCATION_KEY]: true,
          [SIDEBAR_REOPEN_DH_SESSION_LOCATION_KEY]: expect.any(Number),
        }),
      }),
    )
  })
})
