import { describe, expect, it, vi } from 'vitest'

vi.mock('@/components/IconFont', () => ({
  default: () => <span data-testid="icon-font" />,
}))

import { getDigitalHumanManagementMenuItems } from './utils'

describe('DigitalHuman/Management/utils', () => {
  it('所有数字员工都提供发起对话入口', () => {
    const onMenuClick = vi.fn()

    const items = getDigitalHumanManagementMenuItems(
      {
        id: 'any-agent',
        name: '任意数字员工',
      } as never,
      {
        isPinnedInSidebar: false,
      },
      onMenuClick,
    )

    expect(items?.map((item) => item?.key)).toEqual(['session', 'pinSidebar', 'edit', 'delete'])
  })
})
